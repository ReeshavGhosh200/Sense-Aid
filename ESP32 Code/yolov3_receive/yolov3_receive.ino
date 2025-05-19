#include <WiFi.h>
#include <WebServer.h>
#include <HardwareSerial.h>

// WiFi Credentials
const char* ssid = "vivo V21e 5g";
const char* password = "sigma69420";

// UART to ESP32 #2 (BT+SD)
#define UART_RX 16  // Connect to TX of ESP32 #2
#define UART_TX 17  // Connect to RX of ESP32 #2
HardwareSerial SerialBT(1);  // UART1

// Web server
WebServer server(80);

// Object detection states
bool busDetected = false;
bool carDetected = false;
bool truckDetected = false;
bool trainDetected = false;
bool bananaDetected = false;
bool bedDetected = false;
bool benchDetected = false;
bool bicycleDetected = false;
bool chairDetected = false;
bool motorcycleDetected = false;
int distance = 200;  // Default distance
int led = 12;

void setup() {
  Serial.begin(115200);
  SerialBT.begin(115200, SERIAL_8N1, UART_RX, UART_TX);
  
  Serial.println("\n[SYSTEM] Starting ESP32 WiFi Receiver...");
  pinMode(led, OUTPUT);
  digitalWrite(led, HIGH);
  WiFi.mode(WIFI_STA);
  Serial.println("[WiFi] Connecting to network: " + String(ssid));
  delay(100);
  WiFi.begin(ssid, password);
  digitalWrite(led, LOW);
  unsigned long startTime = millis();
  
  while (WiFi.status() != WL_CONNECTED && millis() - startTime < 15000) {
    delay(500);
    Serial.print(".");
    digitalWrite(led, HIGH);
  }
  digitalWrite(led, LOW);
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected successfully!");
    Serial.print("[WiFi] IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[WiFi] Connection failed!");
    while(1); // Halt if WiFi fails
  }

  // Set up server endpoints
  server.on("/update", HTTP_GET, handleUpdate);
  server.on("/", HTTP_GET, []() {
    server.send(200, "text/plain", "ESP32 Object Detection Server");
    Serial.println("[HTTP] Received root request");
  });

  server.begin();
  Serial.println("[HTTP] Server started on port 80");
}

void loop() {
  server.handleClient();
  
  // Send commands to Bluetooth ESP32 based on detection
  if (busDetected) {
    SerialBT.println("1");
    Serial.println("[UART] Sent command '1' (bus.wav) to BT ESP32");
  }
  else if (carDetected) {
    SerialBT.println("3");
    Serial.println("[UART] Sent command '3' (car.wav) to BT ESP32");
  }
  else if (truckDetected) {
    SerialBT.println("5");
    Serial.println("[UART] Sent command '5' (truck.wav) to BT ESP32");
  }
  else if (trainDetected) {
    SerialBT.println("6");
    Serial.println("[UART] Sent command '6' (train.wav) to BT ESP32");
  }
  else if (bananaDetected) {
    SerialBT.println("8");
    Serial.println("[UART] Sent command '8' (banana.wav) to BT ESP32");
  }
  else if (bedDetected) {
    if (distance > 100) {
      SerialBT.println("9");
      Serial.println("[UART] Sent command '9' (bed_far.wav) to BT ESP32");
    } else {
      SerialBT.println("10");
      Serial.println("[UART] Sent command '10' (bed_near.wav) to BT ESP32");
    }
  }
  else if (benchDetected) {
    if (distance > 100) {
      SerialBT.println("11");
      Serial.println("[UART] Sent command '11' (bench_far.wav) to BT ESP32");
    } else {
      SerialBT.println("12");
      Serial.println("[UART] Sent command '12' (bench_near.wav) to BT ESP32");
    }
  }
  else if (bicycleDetected) {
    SerialBT.println("13");
    Serial.println("[UART] Sent command '13' (bicycle.wav) to BT ESP32");
  }
  else if (chairDetected) {
    if (distance > 100) {
      SerialBT.println("15");
      Serial.println("[UART] Sent command '15' (chair_far.wav) to BT ESP32");
    } else {
      SerialBT.println("14");
      Serial.println("[UART] Sent command '14' (chair_near.wav) to BT ESP32");
    }
  }
  else if (motorcycleDetected) {
    SerialBT.println("16");
    Serial.println("[UART] Sent command '16' (motorcycle.wav) to BT ESP32");
  }
  else {
    SerialBT.println("0");
    Serial.println("[UART] Sent command '0' (stop) to BT ESP32");
  }

  delay(100);
}

void handleUpdate() {
  Serial.println("\n[HTTP] Received detection update");
  
  // Update distance if provided
  if (server.hasArg("distance")) {
    distance = server.arg("distance").toInt();
    Serial.print("[DETECTION] Distance updated: ");
    Serial.println(distance);
  }

  // Update detection states
  busDetected = server.hasArg("bus") && server.arg("bus") == "1";
  carDetected = server.hasArg("car") && server.arg("car") == "1";
  truckDetected = server.hasArg("truck") && server.arg("truck") == "1";
  trainDetected = server.hasArg("train") && server.arg("train") == "1";
  bananaDetected = server.hasArg("banana") && server.arg("banana") == "1";
  bedDetected = server.hasArg("bed") && server.arg("bed") == "1";
  benchDetected = server.hasArg("bench") && server.arg("bench") == "1";
  bicycleDetected = server.hasArg("bicycle") && server.arg("bicycle") == "1";
  chairDetected = server.hasArg("chair") && server.arg("chair") == "1";
  motorcycleDetected = server.hasArg("motorcycle") && server.arg("motorcycle") == "1";

  // Log current detections
  Serial.println("[DETECTION] Current states:");
  Serial.print("  Bus: "); Serial.println(busDetected ? "DETECTED" : "Not detected");
  Serial.print("  Car: "); Serial.println(carDetected ? "DETECTED" : "Not detected");
  Serial.print("  Truck: "); Serial.println(truckDetected ? "DETECTED" : "Not detected");
  Serial.print("  Train: "); Serial.println(trainDetected ? "DETECTED" : "Not detected");
  Serial.print("  Banana: "); Serial.println(bananaDetected ? "DETECTED" : "Not detected");
  Serial.print("  Bed: "); Serial.println(bedDetected ? "DETECTED" : "Not detected");
  Serial.print("  Bench: "); Serial.println(benchDetected ? "DETECTED" : "Not detected");
  Serial.print("  Bicycle: "); Serial.println(bicycleDetected ? "DETECTED" : "Not detected");
  Serial.print("  Chair: "); Serial.println(chairDetected ? "DETECTED" : "Not detected");
  Serial.print("  Motorcycle: "); Serial.println(motorcycleDetected ? "DETECTED" : "Not detected");
  Serial.print("  Distance: "); Serial.print(distance); Serial.println(" units");

  server.send(200, "text/plain", "Detection states updated");
}