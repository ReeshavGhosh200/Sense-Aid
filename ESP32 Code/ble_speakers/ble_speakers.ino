#include <Arduino.h>
#include <BluetoothA2DPSource.h>
#include <SD.h>
#include <SPI.h>

// SD Card pins
#define SD_CS 5
#define SPI_MOSI 23
#define SPI_MISO 19
#define SPI_SCK 18

// Target device MAC address
// Example-     33:33:22:39:73:13
uint8_t target_address[6] = {0x33, 0x33, 0x22, 0x39, 0x73, 0x13};

// A2DP Source instance
BluetoothA2DPSource a2dp_source;
bool connected = false;
bool shouldPlay = false;
bool shouldStop = false;
File audioFile;
String currentFile = "";

// Audio callback function
int32_t get_data_callback(uint8_t *data, int32_t len) {
  if (!shouldPlay || !audioFile) {
    return 0;
  }
  
  size_t bytes_read = audioFile.read(data, len);
  
  if (bytes_read < len || shouldStop) {
    shouldPlay = false;
    audioFile.close();
    currentFile = "";
    Serial.println("Playback ended");
    if (shouldStop) {
      shouldStop = false;
      return 0;
    }
  }
  
  return bytes_read;
}

void connection_state_changed(esp_a2d_connection_state_t state, void *) {
  if (state == ESP_A2D_CONNECTION_STATE_CONNECTED) {
    Serial.println("Bluetooth connected");
    connected = true;
  } else {
    Serial.println("Bluetooth disconnected");
    connected = false;
    shouldPlay = false;
    shouldStop = false;
    if (audioFile) audioFile.close();
    currentFile = "";
  }
}

bool parseWavHeader() {
  uint8_t wav_header[44];
  audioFile.seek(0);
  audioFile.read(wav_header, 44);
  
  if (wav_header[0] != 'R' || wav_header[1] != 'I' || wav_header[2] != 'F' || wav_header[3] != 'F') {
    Serial.println("Invalid WAV file");
    return false;
  }
  
  audioFile.seek(44);
  return true;
}

bool initSD() {
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  if (!SD.begin(SD_CS)) {
    Serial.println("SD card failed");
    return false;
  }
  Serial.println("SD card ready");
  return true;
}

void initBluetooth() {
  a2dp_source.set_on_connection_state_changed(connection_state_changed);
  a2dp_source.set_data_callback(get_data_callback);
  a2dp_source.start("ESP32-A2DP-Source");
  a2dp_source.connect_to(target_address);
  Serial.println("Bluetooth initialized");
}

void stopPlayback() {
  shouldStop = true;
  Serial.println("Stop command received - will stop after current file");
}

void playFile(const char* filePath) {
  // Don't restart the same file
  if (currentFile == filePath && shouldPlay) {
    Serial.println("File already playing");
    return;
  }
  
  // Stop any current playback
  shouldStop = true;
  while (shouldPlay) {
    delay(10); // Wait for current playback to stop
  }

  if (!connected) {
    Serial.println("Not connected, attempting to connect...");
    a2dp_source.connect_to(target_address);
    delay(1000);
    if (!connected) {
      Serial.println("Still not connected");
      return;
    }
  }

  if (!SD.exists(filePath)) {
    Serial.print("File not found: ");
    Serial.println(filePath);
    return;
  }

  audioFile = SD.open(filePath);
  if (!audioFile) {
    Serial.println("Failed to open file");
    return;
  }

  if (!parseWavHeader()) {
    audioFile.close();
    return;
  }

  currentFile = filePath;
  shouldPlay = true;
  shouldStop = false;
  Serial.print("Now playing: ");
  Serial.println(filePath);
}

void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, 16, 17);
  
  Serial.println("\nStarting Audio Player...");
  
  if (!initSD()) {
    Serial.println("SD card failed, halting");
    while(1);
  }
  
  initBluetooth();
  
  Serial.println("System ready. Commands:");
  Serial.println("1-16: Play corresponding audio");
  Serial.println("0: Stop playback after current file");
}

void loop() {
  // Handle Serial input
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    processCommand(input);
  }
  
  // Handle Serial2 input
  if (Serial2.available()) {
    String input = Serial2.readStringUntil('\n');
    input.trim();
    processCommand(input);
  }
  
  // Reconnect Bluetooth if needed
  if (!connected && millis() % 5000 < 100) {
    a2dp_source.connect_to(target_address);
  }
}

void processCommand(String input) {
  if (input == "0") {
    stopPlayback();
    return;
  }
  
  const char* fileMap[17] = {
    nullptr,
    "/bus.wav",       // 1
    "/buses.wav",     // 2
    "/car.wav",       // 3
    "/cars.wav",      // 4
    "/truck.wav",     // 5
    "/train.wav",     // 6
    "/trains.wav",    // 7
    "/banana.wav",    // 8
    "/bed_far.wav",   // 9
    "/bed_near.wav",  // 10
    "/bench_far.wav", // 11
    "/bench_near.wav",// 12
    "/bicycle.wav",   // 13
    "/chair_near.wav",// 14
    "/chair_far.wav", // 15
    "/motorcycle.wav" // 16
  };

  int command = input.toInt();
  if (command >= 1 && command <= 16) {
    playFile(fileMap[command]);
  } else {
    Serial.println("Invalid command (0-16)");
  }
}
