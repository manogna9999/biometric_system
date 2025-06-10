const int ecgPin = A0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int ecgValue = analogRead(ecgPin);
  Serial.println(ecgValue);
  delay(10);
}

