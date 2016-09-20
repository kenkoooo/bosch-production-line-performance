import java.io.*;

class Utils {
  static BufferedReader getBufferedReader(String filename) throws FileNotFoundException {
    File file = new File(filename);
    InputStream csvStream = new FileInputStream(file);
    return new BufferedReader(new InputStreamReader(csvStream));
  }

  static PrintWriter getPrintWriter(String filename) throws IOException {
    File file = new File(filename);
    if (!file.exists()) file.createNewFile();
    return new PrintWriter(new BufferedWriter(new FileWriter(file)));
  }

  static void elapsedTime(long start) {
    System.out.println(((System.currentTimeMillis() - start) / 1000.0) + " sec");
  }

  static int getFullSize(String filename) throws IOException {
    BufferedReader br = getBufferedReader(filename);
    int count = 0;
    while (br.readLine() != null) {
      count++;
    }
    return count;
  }

}
