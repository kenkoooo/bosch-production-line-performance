import java.io.*;
import java.util.zip.GZIPInputStream;

class Utils {
  static BufferedReader getBufferedReader(String filename) throws FileNotFoundException {
    File file = new File(filename);
    InputStream csvStream = new FileInputStream(file);
    return new BufferedReader(new InputStreamReader(csvStream));
  }

  static BufferedReader getGZBufferedReader(String filename) throws IOException {
    File file = new File(filename);
    InputStream csvStream = new FileInputStream(file);
    return new BufferedReader(new InputStreamReader(new GZIPInputStream(new BufferedInputStream(csvStream))));
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
    return getFullSize(filename, false);
  }

  static int getFullSize(String filename, boolean isGZ) throws IOException {
    BufferedReader br;
    if (isGZ) {
      br = getGZBufferedReader(filename);
    } else {
      br = getBufferedReader(filename);
    }
    int count = 0;
    while (br.readLine() != null) {
      count++;
    }
    return count;
  }
}
