import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

public class Binarizer {
  private final long start = System.currentTimeMillis();
  private String input = "./output/reduced_train_categorical.csv.gz";
  private String output = "./output/reduced_train_categorical_binary.csv";

  private void run() throws IOException {
    BufferedReader reader = Utils.getGZBufferedReader(input);
    PrintWriter writer = Utils.getPrintWriter(output);

    String line = reader.readLine();
    String[] header = line.split(",");
    for (int i = 1; i < header.length; i++) {
      header[i] = header[i] + "_BIN";
    }
    for (int i = 0; i < header.length; i++) {
      if (i > 0) writer.print(",");
      writer.print(header[i]);
    }
    writer.println();

    int N = header.length;
    int[] bin = new int[N];
    int count = 0;
    while ((line = reader.readLine()) != null) {
      String[] row = line.split(",");
      Arrays.fill(bin, 0);
      for (int i = 0; i < row.length; i++) {
        if (!row[i].equals("")) bin[i] = 1;
      }
      for (int i = 0; i < N; i++) {
        if (i > 0) {
          writer.print(",");
          writer.print(bin[i]);
        } else {
          writer.print(row[i]);
        }
      }
      writer.println();

      count++;
      if (count % 10000 == 0) {
        Utils.elapsedTime(start);
        System.out.println(count);
      }
    }
    writer.close();
  }

  public static void main(String[] args) throws IOException {
    Binarizer maker = new Binarizer();
    maker.run();
    Utils.elapsedTime(maker.start);
  }
}
