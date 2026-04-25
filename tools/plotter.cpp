#include "plotter.hpp"

#include <arpa/inet.h>   // htons, inet_addr
#include <sys/socket.h>  // socket, sendto
#include <unistd.h>      // close

#include <yaml-cpp/yaml.h>

namespace tools
{
Plotter::Plotter(std::string host, uint16_t port)
{
  socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);

  setDestination(host, port);
}

Plotter::~Plotter() { ::close(socket_); }

void Plotter::configure(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  const auto host = yaml["plotter_host"].as<std::string>("127.0.0.1");
  const auto port = yaml["plotter_port"].as<uint16_t>(9870);

  std::lock_guard<std::mutex> lock(mutex_);
  setDestination(host, port);
}

void Plotter::setDestination(const std::string & host, uint16_t port)
{
  destination_.sin_family = AF_INET;
  destination_.sin_port = ::htons(port);
  destination_.sin_addr.s_addr = ::inet_addr(host.c_str());
}

void Plotter::plot(const nlohmann::json & json)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto data = json.dump();
  ::sendto(
    socket_, data.c_str(), data.length(), 0, reinterpret_cast<sockaddr *>(&destination_),
    sizeof(destination_));
}

}  // namespace tools