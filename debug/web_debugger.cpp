#include "web_debugger.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

// SHA-1 + Base64 for WebSocket handshake (self-contained, no OpenSSL needed)
#include <array>
#include <cstdint>

namespace debug
{

namespace
{
void set_socket_timeout(int fd, int sec, int usec = 0)
{
  timeval tv{sec, usec};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}
}

// ── SHA-1 (RFC 3174) ─────────────────────────────────────────────────────────
static std::array<uint8_t, 20> sha1(const std::string & input)
{
  uint32_t h[5] = {0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0};
  std::vector<uint8_t> msg(input.begin(), input.end());
  uint64_t bit_len = msg.size() * 8;
  msg.push_back(0x80);
  while (msg.size() % 64 != 56) msg.push_back(0);
  for (int i = 7; i >= 0; --i) msg.push_back((bit_len >> (i * 8)) & 0xFF);

  for (size_t i = 0; i < msg.size(); i += 64) {
    uint32_t w[80];
    for (int j = 0; j < 16; ++j)
      w[j] = (msg[i+j*4]<<24)|(msg[i+j*4+1]<<16)|(msg[i+j*4+2]<<8)|msg[i+j*4+3];
    for (int j = 16; j < 80; ++j) {
      uint32_t v = w[j-3]^w[j-8]^w[j-14]^w[j-16];
      w[j] = (v<<1)|(v>>31);
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4];
    for (int j = 0; j < 80; ++j) {
      uint32_t f,k;
      if      (j<20){f=(b&c)|((~b)&d);k=0x5A827999;}
      else if (j<40){f=b^c^d;         k=0x6ED9EBA1;}
      else if (j<60){f=(b&c)|(b&d)|(c&d);k=0x8F1BBCDC;}
      else          {f=b^c^d;         k=0xCA62C1D6;}
      uint32_t tmp=((a<<5)|(a>>27))+f+e+k+w[j];
      e=d;d=c;c=(b<<30)|(b>>2);b=a;a=tmp;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;
  }
  std::array<uint8_t,20> out;
  for (int i=0;i<5;++i) for (int j=3;j>=0;--j) out[i*4+(3-j)]=(h[i]>>(j*8))&0xFF;
  return out;
}

static std::string base64_encode(const uint8_t * data, size_t len)
{
  static const char * t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (size_t i = 0; i < len; i += 3) {
    uint32_t v = (uint32_t)data[i] << 16;
    if (i+1 < len) v |= (uint32_t)data[i+1] << 8;
    if (i+2 < len) v |= data[i+2];
    out += t[(v>>18)&63]; out += t[(v>>12)&63];
    out += (i+1<len) ? t[(v>>6)&63] : '=';
    out += (i+2<len) ? t[v&63]      : '=';
  }
  return out;
}

// ── WebDebugger ───────────────────────────────────────────────────────────────
WebDebugger::WebDebugger(int port) : port_(port) {}

WebDebugger::~WebDebugger() { stop(); }

void WebDebugger::start()
{
  running_ = true;
  server_thread_ = std::thread(&WebDebugger::serve, this);
}

void WebDebugger::stop()
{
  running_ = false;
  if (server_thread_.joinable()) server_thread_.join();
}

void WebDebugger::push(
  const cv::Mat & frame,
  const std::vector<DetectionData> & detections,
  const std::vector<ReprojectionData> & reprojections,
  double latency_ms)
{
  std::string json = build_json(frame, detections, reprojections, latency_ms);
  {
    std::lock_guard<std::mutex> lk(data_mutex_);
    latest_json_ = json;
  }
  broadcast(ws_frame(json));
}

// ── HTTP + WebSocket 服务主循环 ───────────────────────────────────────────────
void WebDebugger::serve()
{
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) return;

  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port_);

  if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) { close(server_fd); return; }
  listen(server_fd, 32);

  // 非阻塞 accept
  fd_set fds;

  while (running_) {
    struct timeval tv{0, 100000};  // select 会修改 timeval，必须每轮重置
    FD_ZERO(&fds);
    FD_SET(server_fd, &fds);
    if (select(server_fd + 1, &fds, nullptr, nullptr, &tv) > 0) {
      int client_fd = accept(server_fd, nullptr, nullptr);
      if (client_fd >= 0) {
        set_socket_timeout(client_fd, 1);
        std::thread([this, client_fd]() { handle_http(client_fd); }).detach();
      }
    }
  }
  close(server_fd);
}

void WebDebugger::handle_http(int fd)
{
  char buf[4096] = {};
  int n = recv(fd, buf, sizeof(buf) - 1, 0);
  if (n <= 0) { close(fd); return; }

  std::string req(buf, n);

  // WebSocket 升级请求
  if (req.find("Upgrade: websocket") != std::string::npos) {
    // 提取 Sec-WebSocket-Key
    auto pos = req.find("Sec-WebSocket-Key: ");
    if (pos == std::string::npos) { close(fd); return; }
    pos += 19;
    auto end = req.find("\r\n", pos);
    std::string key = req.substr(pos, end - pos);

    std::string resp = ws_handshake_response(key);
    if (send(fd, resp.c_str(), resp.size(), MSG_NOSIGNAL) < 0) {
      close(fd);
      return;
    }

    {
      std::lock_guard<std::mutex> lk(clients_mutex_);
      ws_clients_.push_back(fd);
    }

    // 保持连接，读取客户端消息（忽略内容，只检测断开）
    handle_websocket(fd);

    {
      std::lock_guard<std::mutex> lk(clients_mutex_);
      ws_clients_.erase(std::remove(ws_clients_.begin(), ws_clients_.end(), fd), ws_clients_.end());
    }
    close(fd);
    return;
  }

  // 普通 HTTP 请求 — 返回前端页面
  std::string path = "/";
  auto p0 = req.find("GET ");
  if (p0 != std::string::npos) {
    auto p1 = req.find(' ', p0 + 4);
    path = req.substr(p0 + 4, p1 - p0 - 4);
  }

  // 读取 index.html
  std::string html_path = std::string(__FILE__);
  html_path = html_path.substr(0, html_path.rfind('/')) + "/static/index.html";

  std::ifstream f(html_path);
  std::string body;
  if (f.good()) {
    body = std::string(std::istreambuf_iterator<char>(f), {});
  } else {
    body = "<h1>SPR_Vision_26 Debugger</h1><p>index.html not found</p>";
  }

  std::string resp =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/html; charset=utf-8\r\n"
    "Content-Length: " + std::to_string(body.size()) + "\r\n"
    "Connection: close\r\n\r\n" + body;

  send(fd, resp.c_str(), resp.size(), 0);
  close(fd);
}

void WebDebugger::handle_websocket(int fd)
{
  // 先推送最新帧
  {
    std::lock_guard<std::mutex> lk(data_mutex_);
    if (!latest_json_.empty()) {
      std::string frame = ws_frame(latest_json_);
      if (send(fd, frame.c_str(), frame.size(), MSG_NOSIGNAL | MSG_DONTWAIT) < 0 &&
          errno != EAGAIN && errno != EWOULDBLOCK) {
        return;
      }
    }
  }

  // 保持连接直到客户端断开
  std::vector<uint8_t> buf(256);
  while (running_) {
    struct timeval tv{1, 0};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    int ready = select(fd + 1, &fds, nullptr, nullptr, &tv);
    if (ready > 0) {
      int n = recv(fd, buf.data(), buf.size(), 0);
      if (n <= 0) break;  // 客户端断开
    } else if (ready < 0) {
      break;
    }
  }
}

void WebDebugger::broadcast(const std::string & msg)
{
  std::lock_guard<std::mutex> lk(clients_mutex_);
  for (auto it = ws_clients_.begin(); it != ws_clients_.end(); ) {
    ssize_t r = send(*it, msg.c_str(), msg.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
    if (r < 0 || static_cast<size_t>(r) != msg.size()) {
      shutdown(*it, SHUT_RDWR);
      it = ws_clients_.erase(it);
    } else {
      ++it;
    }
  }
}

// ── 工具函数 ──────────────────────────────────────────────────────────────────
std::string WebDebugger::mat_to_base64_jpg(const cv::Mat & img, int quality)
{
  std::vector<uint8_t> buf;
  cv::imencode(".jpg", img, buf, {cv::IMWRITE_JPEG_QUALITY, quality});
  return base64_encode(buf.data(), buf.size());
}

std::string WebDebugger::build_json(
  const cv::Mat & frame,
  const std::vector<DetectionData> & detections,
  const std::vector<ReprojectionData> & reprojections,
  double latency_ms)
{
  // 手写 JSON，避免引入 nlohmann_json 依赖
  std::ostringstream ss;
  ss << "{";

  // frame (base64 jpg，缩小到 640 宽以减少带宽)
  cv::Mat small;
  float scale = 1.0f;
  if (frame.cols > 640) {
    scale = 640.0f / frame.cols;
    cv::resize(frame, small, {}, scale, scale);
  } else {
    small = frame;
  }
  ss << "\"frame\":\"" << mat_to_base64_jpg(small) << "\",";

  // detections
  ss << "\"detections\":[";
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto & d = detections[i];
    if (i) ss << ",";
    ss << "{\"pts\":[";
    for (size_t j = 0; j < d.pts.size(); ++j) {
      if (j) ss << ",";
      ss << "[" << d.pts[j].x * scale << "," << d.pts[j].y * scale << "]";
    }
    ss << "],\"color\":" << d.color
       << ",\"number\":" << d.number
       << ",\"conf\":" << d.conf << "}";
  }
  ss << "],";

  // reprojections
  ss << "\"reprojections\":[";
  for (size_t i = 0; i < reprojections.size(); ++i) {
    const auto & r = reprojections[i];
    if (i) ss << ",";
    ss << "{\"pts\":[";
    for (size_t j = 0; j < r.pts.size(); ++j) {
      if (j) ss << ",";
      ss << "[" << r.pts[j].x * scale << "," << r.pts[j].y * scale << "]";
    }
    ss << "]}";
  }
  ss << "],";

  ss << "\"latency_ms\":" << latency_ms;
  ss << "}";
  return ss.str();
}

std::string WebDebugger::ws_handshake_response(const std::string & key)
{
  static const std::string GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
  std::string combined = key + GUID;
  auto digest = sha1(combined);
  std::string accept = base64_encode(digest.data(), digest.size());

  return "HTTP/1.1 101 Switching Protocols\r\n"
         "Upgrade: websocket\r\n"
         "Connection: Upgrade\r\n"
         "Sec-WebSocket-Accept: " + accept + "\r\n\r\n";
}

std::string WebDebugger::ws_frame(const std::string & payload)
{
  // RFC 6455 WebSocket text frame，服务端发送不需要 masking
  std::string frame;
  frame += (char)0x81;  // FIN + text opcode

  size_t len = payload.size();
  if (len <= 125) {
    frame += (char)len;
  } else if (len <= 65535) {
    frame += (char)126;
    frame += (char)((len >> 8) & 0xFF);
    frame += (char)(len & 0xFF);
  } else {
    frame += (char)127;
    for (int i = 7; i >= 0; --i) frame += (char)((len >> (i * 8)) & 0xFF);
  }
  frame += payload;
  return frame;
}

bool WebDebugger::parse_ws_frame(const std::vector<uint8_t> & buf, std::string & out)
{
  if (buf.size() < 2) return false;
  // 简单解析，仅用于检测 close frame
  uint8_t opcode = buf[0] & 0x0F;
  if (opcode == 0x8) return false;  // close
  return true;
}

}  // namespace debug
