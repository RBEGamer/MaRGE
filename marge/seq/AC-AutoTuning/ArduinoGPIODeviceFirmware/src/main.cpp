#include <Arduino.h>

namespace {

constexpr uint8_t kPins[] = {
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, A0, A1, A2, A3, A4, A5,
};

constexpr uint8_t kPinModes[] = {
    OUTPUT, OUTPUT, OUTPUT, OUTPUT, OUTPUT, OUTPUT, OUTPUT, OUTPUT, OUTPUT,
    OUTPUT, OUTPUT, OUTPUT, OUTPUT, OUTPUT, OUTPUT, OUTPUT, OUTPUT, OUTPUT,
};

void printPinLabel(uint8_t pin) {
  Serial.print((pin >= A0) ? 'A' : 'D');
  Serial.print((pin >= A0) ? pin - A0 : pin);
}

String readCommand() {
  static String buffer;

  while (Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\r') {
      continue;
    }

    if (ch == '\n') {
      String command = buffer;
      buffer = "";
      command.trim();
      command.toUpperCase();
      return command;
    }

    if (buffer.length() < 63) {
      buffer += ch;
    }
  }

  return "";
}

bool parsePinToken(const String& token, uint8_t& pin) {
  if (token.length() < 2) {
    return false;
  }

  if (token[0] == 'D') {
    const int value = token.substring(1).toInt();
    if (value >= 2 && value <= 13) {
      pin = static_cast<uint8_t>(value);
      return true;
    }
    return false;
  }

  if (token[0] == 'A') {
    const int value = token.substring(1).toInt();
    if (value >= 0 && value <= 5) {
      pin = static_cast<uint8_t>(A0 + value);
      return true;
    }
    return false;
  }

  return false;
}

int findPinIndex(uint8_t pin) {
  for (size_t index = 0; index < (sizeof(kPins) / sizeof(kPins[0])); ++index) {
    if (kPins[index] == pin) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

bool isWritablePin(uint8_t pin) {
  const int index = findPinIndex(pin);
  return index >= 0 && kPinModes[index] == OUTPUT;
}

void writeAllPins(uint8_t level) {
  for (size_t index = 0; index < (sizeof(kPins) / sizeof(kPins[0])); ++index) {
    if (kPinModes[index] == OUTPUT) {
      digitalWrite(kPins[index], level);
    }
  }
}

const __FlashStringHelper* pinModeName(uint8_t mode) {
  if (mode == OUTPUT) {
    return F("OUTPUT");
  }
  if (mode == INPUT) {
    return F("INPUT");
  }
  if (mode == INPUT_PULLUP) {
    return F("INPUT_PULLUP");
  }
  return F("UNKNOWN");
}

void printStatus() {
  Serial.print(F("STATUS"));

  for (const uint8_t pin : kPins) {
    Serial.print(' ');
    printPinLabel(pin);
    Serial.print('=');
    Serial.print(digitalRead(pin) == HIGH ? F("HIGH") : F("LOW"));
  }

  Serial.println();
}

void printPins() {
  Serial.print(F("PINS"));

  for (size_t index = 0; index < (sizeof(kPins) / sizeof(kPins[0])); ++index) {
    Serial.print(' ');
    printPinLabel(kPins[index]);
    Serial.print('=');
    Serial.print(pinModeName(kPinModes[index]));
  }

  Serial.println();
}

void printHelp() {
  Serial.println(F("Commands:"));
  Serial.println(F("  SET D2 HIGH"));
  Serial.println(F("  SET D13 LOW"));
  Serial.println(F("  SET A0 HIGH"));
  Serial.println(F("  ALL HIGH"));
  Serial.println(F("  ALL LOW"));
  Serial.println(F("  PINS"));
  Serial.println(F("  STATUS"));
  Serial.println(F("  HELP"));
}

void handleSetCommand(const String& command) {
  const int firstSpace = command.indexOf(' ');
  const int secondSpace = command.indexOf(' ', firstSpace + 1);

  if (firstSpace < 0 || secondSpace < 0) {
    Serial.println(F("ERR invalid SET syntax"));
    return;
  }

  const String pinToken = command.substring(firstSpace + 1, secondSpace);
  const String levelToken = command.substring(secondSpace + 1);

  uint8_t pin = 0;
  if (!parsePinToken(pinToken, pin)) {
    Serial.println(F("ERR unsupported pin"));
    return;
  }

  if (!isWritablePin(pin)) {
    Serial.println(F("ERR pin is not writable"));
    return;
  }

  if (levelToken == "HIGH") {
    digitalWrite(pin, HIGH);
    Serial.println(F("OK"));
    return;
  }

  if (levelToken == "LOW") {
    digitalWrite(pin, LOW);
    Serial.println(F("OK"));
    return;
  }

  Serial.println(F("ERR level must be HIGH or LOW"));
}

void handleCommand(const String& command) {
  if (command.length() == 0) {
    return;
  }

  if (command == "HELP") {
    printHelp();
    return;
  }

  if (command == "STATUS") {
    printStatus();
    return;
  }

  if (command == "PINS") {
    printPins();
    return;
  }

  if (command == "ALL HIGH") {
    writeAllPins(HIGH);
    Serial.println(F("OK"));
    return;
  }

  if (command == "ALL LOW") {
    writeAllPins(LOW);
    Serial.println(F("OK"));
    return;
  }

  if (command.startsWith("SET ")) {
    handleSetCommand(command);
    return;
  }

  Serial.println(F("ERR unknown command"));
}

}  // namespace

void setup() {
  for (size_t index = 0; index < (sizeof(kPins) / sizeof(kPins[0])); ++index) {
    const uint8_t pin = kPins[index];
    pinMode(pin, kPinModes[index]);
    digitalWrite(pin, LOW);
  }

  Serial.begin(115200);
  while (!Serial) {
  }

  Serial.println(F("UNO GPIO serial controller ready"));
  printHelp();
}

void loop() {
  const String command = readCommand();
  if (command.length() > 0) {
    handleCommand(command);
  }
}
