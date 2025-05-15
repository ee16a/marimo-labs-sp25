unsigned long firstSensor = 0;      /* Accumulator for first sensor ADC readings */
unsigned long secondSensor = 0;     /* Accumulator for second sensor ADC readings */
unsigned int numAvgs = 500;         /* Number of averages to take for each sensor per sensor reading */
int handshake = 0;                  /* Intermediate storage for input character */

void setup() {
  /* Start serial connection at 115200 baud */
  Serial.begin(115200);
  while (!Serial) {
    ; /* Wait for serial port to connect. Needed for Leonardo only. */
  }
}

void loop() {
  /* If we get a valid byte, read analog inputs */
  if (Serial.available() > 0) {
    handshake = Serial.read();
    if (handshake == 57) {
      /* On ASCII character '9' (0d57), flush the terminal */
      Serial.flush();
    } else if (handshake == 54) {
      /* On ASCII character '6' (0d54), read the sensor */
      /* Reset sensor accumulators and take numAvgs readings of each sensor independently */
      firstSensor = 0;
      secondSensor = 0;
      for (int count = 0; count < numAvgs; count++) {
        firstSensor += analogRead(A0);
        secondSensor += analogRead(A1);
      }
      /* Average numAvgs readings between both sensors and print to serial */
      Serial.println((firstSensor + secondSensor) / (2 * numAvgs), DEC);
    }
  }
}
