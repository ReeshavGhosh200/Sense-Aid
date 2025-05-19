// #include <Arduino.h>
// #include <BluetoothA2DPSource.h>
// #include <SD.h>
// #include <SPI.h>

// // SD Card pins - using default SPI pins for ESP32
// #define SD_CS 5      // SD Card chip select pin
// #define SPI_MOSI 23  // SD Card MOSI pin
// #define SPI_MISO 19  // SD Card MISO pin
// #define SPI_SCK 18   // SD Card SCK pin

// // Target device MAC address: 34:6e:28:b6:df:c1
// // 41:42:60:ab:db:4c
// // 33:33:22:39:73:13
// // uint8_t target_address[6] = {0x34, 0x6e, 0x28, 0xb6, 0xdf, 0xc1};
// uint8_t target_address[6] = {0x33, 0x33, 0x22, 0x39, 0x73, 0x13};

// // Buffer for reading from SD card
// #define BUFFER_SIZE 512
// uint8_t audio_buffer[BUFFER_SIZE];

// // A2DP Source instance
// BluetoothA2DPSource a2dp_source;
// bool connected = false;
// bool playing = false;
// bool playback_complete = false;

// // WAV file settings
// File audioFile;
// uint8_t wav_header[44]; // WAV header is 44 bytes
// int32_t sample_rate;
// uint16_t num_channels;
// uint16_t bits_per_sample;

// // Audio data callback function
// int32_t get_data_callback(uint8_t *data, int32_t len) {
//   if (!playing || !audioFile) {
//     return 0;
//   }
  
//   size_t bytes_read = audioFile.read(data, len);
  
//   if (bytes_read < len) {
//     playback_complete = true;
//     return bytes_read;
//   }
  
//   return bytes_read;
// }

// // Callback function for connection state changes
// void connection_state_changed(esp_a2d_connection_state_t state, void *) {
//   if (state == ESP_A2D_CONNECTION_STATE_CONNECTED) {
//     Serial.println("Connected to target device");
//     connected = true;
//   } else if (state == ESP_A2D_CONNECTION_STATE_DISCONNECTED) {
//     Serial.println("Disconnected from target device");
//     connected = false;
//   }
// }

// // Parse WAV header to get audio format details
// bool parseWavHeader() {
//   audioFile.seek(0);
//   audioFile.read(wav_header, 44);
  
//   if (wav_header[0] != 'R' || wav_header[1] != 'I' || 
//       wav_header[2] != 'F' || wav_header[3] != 'F') {
//     Serial.println("Not a valid WAV file");
//     return false;
//   }
  
//   sample_rate = wav_header[24] | (wav_header[25] << 8) | 
//                 (wav_header[26] << 16) | (wav_header[27] << 24);
  
//   num_channels = wav_header[22] | (wav_header[23] << 8);
//   bits_per_sample = wav_header[34] | (wav_header[35] << 8);
  
//   Serial.print("WAV File: ");
//   Serial.print(sample_rate);
//   Serial.print("Hz, ");
//   Serial.print(bits_per_sample);
//   Serial.print(" bits, ");
//   Serial.print(num_channels);
//   Serial.println(" channels");
  
//   audioFile.seek(44);
//   return true;
// }

// // Initialize SD card
// bool initSD() {
//   SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  
//   if (!SD.begin(SD_CS)) {
//     Serial.println("SD card initialization failed!");
//     return false;
//   }
  
//   Serial.println("SD card initialized.");
//   return true;
// }

// // Initialize Bluetooth
// void initBluetooth() {
//   a2dp_source.set_on_connection_state_changed(connection_state_changed);
//   a2dp_source.set_data_callback(get_data_callback);
//   a2dp_source.start("ESP32-A2DP-Source");
//   Serial.println("Connecting to device: SW-0049 (34:6e:28:b6:df:c1)...");
//   a2dp_source.connect_to(target_address);
// }

// // Play function that blocks until playback is complete
// void play(const char* filePath) {
//   Serial.print("Playing file: ");
//   Serial.println(filePath);
  
//   if (!connected) {
//     Serial.println("Not connected to Bluetooth device. Attempting to connect...");
//     a2dp_source.connect_to(target_address);
    
//     unsigned long startTime = millis();
//     while (!connected && (millis() - startTime < 10000)) {
//       delay(100);
//     }
    
//     if (!connected) {
//       Serial.println("Failed to connect to Bluetooth device. Cannot play file.");
//       return;
//     }
//   }
  
//   if (!SD.exists(filePath)) {
//     Serial.print("Error: File not found - ");
//     Serial.println(filePath);
//     return;
//   }
  
//   if (audioFile) {
//     audioFile.close();
//   }
  
//   audioFile = SD.open(filePath);
//   if (!audioFile) {
//     Serial.println("Failed to open audio file!");
//     return;
//   }
  
//   if (!parseWavHeader()) {
//     Serial.println("Failed to parse WAV header. Cannot play file.");
//     audioFile.close();
//     return;
//   }
  
//   playback_complete = false;
//   playing = true;
  
//   Serial.println("Playback started");
  
//   while (!playback_complete && connected) {
//     delay(100);
//   }
  
//   playing = false;
  
//   if (playback_complete) {
//     Serial.println("Playback completed");
//   } else {
//     Serial.println("Playback interrupted: Bluetooth disconnected");
//   }
  
//   audioFile.close();
// }

// void setup() {
//   Serial.begin(115200);
//   Serial2.begin(115200, SERIAL_8N1, 16, 17);  // UART2 on GPIO 16 (RX2), 17 (TX2)
//   Serial.println("ESP32 Bluetooth A2DP Source with SD Card Audio Player");
  
//   if (!initSD()) {
//     Serial.println("SD card setup failed. Cannot continue.");
//     while (1) delay(1000);
//   }
  
//   initBluetooth();
  
//   Serial.println("Setup complete. Ready to play audio files.");
//   Serial.println("Available commands:");
//   Serial.println("1: /bus.wav");
//   Serial.println("2: /buses.wav");
//   Serial.println("3: /car.wav");
//   Serial.println("4: /cars.wav");
//   Serial.println("5: /truck.wav");
//   Serial.println("6: /train.wav");
//   Serial.println("7: /trains.wav");
//   Serial.println("8: /banana.wav");
//   Serial.println("9: /bed_far.wav");
//   Serial.println("10: /bed_near.wav");
//   Serial.println("11: /bench_far.wav");
//   Serial.println("12: /bench_near.wav");
//   Serial.println("13: /bicycle.wav");
//   Serial.println("14: /chair_near.wav");
//   Serial.println("15: /chair_far.wav");
//   Serial.println("16: /motorcycle.wav");
// }

// void loop() {
//   if (Serial.available() > 0) {
//     handleSerialInput(Serial.readStringUntil('\n'));
//   }
  
//   // Check Serial2 (UART2 - GPIO 16/17)
//   if (Serial2.available() > 0) {
//     handleSerialInput(Serial2.readStringUntil('\n'));
//   }
  
//   // Bluetooth connection check
//   if (!connected) {
//     a2dp_source.connect_to(target_address);
//     delay(5000);
//   }
// }

// void handleSerialInput(String input) {
//   input.trim();
//   // if (input == "1") play("/bus.wav");
//   // else if (input == "2") play("/buses.wav");
//   // // ... (rest of the number mappings)
//   // else Serial.println("Invalid input (1-16)");

//   if (input == "1") {
//       play("/bus.wav");
//     } 
//     else if (input == "2") {
//       play("/buses.wav");
//     }
//     else if (input == "3") {
//       play("/car.wav");
//     }
//     else if (input == "4") {
//       play("/cars.wav");
//     }
//     else if (input == "5") {
//       play("/truck.wav");
//     }
//     else if (input == "6") {
//       play("/train.wav");
//     }
//     else if (input == "7") {
//       play("/trains.wav");
//     }
//     else if (input == "8") {
//       play("/banana.wav");
//     }
//     else if (input == "9") {
//       play("/bed_far.wav");
//     }
//     else if (input == "10") {
//       play("/bed_near.wav");
//     }
//     else if (input == "11") {
//       play("/bench_far.wav");
//     }
//     else if (input == "12") {
//       play("/bench_near.wav");
//     }
//     else if (input == "13") {
//       play("/bicycle.wav");
//     }
//     else if (input == "14") {
//       play("/chair_near.wav");
//     }
//     else if (input == "15") {
//       play("/chair_far.wav");
//     }
//     else if (input == "16") {
//       play("/motorcycle.wav");
//     }
//     else {
//       Serial.println("Invalid input. Send numbers 1-16");
//     }

// }

// //if (Serial.available() > 0) {
//   //   String input = Serial.readStringUntil('\n');
//   //   input.trim();
    
    
//   // }
  
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