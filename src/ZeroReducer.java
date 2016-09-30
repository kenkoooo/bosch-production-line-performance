import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;

public class ZeroReducer {
  long start = System.currentTimeMillis();
  String input = "./output/reduced_train_categorical_binary.csv.gz_reduced.csv.gz";
  String output = "./output/reduced_train_categorical_binary.csv.gz_reduced.csv";
  public static void main(String[] args) throws IOException {
    ZeroReducer reducer = new ZeroReducer();
    reducer.run();
  }

  private void run() throws IOException {
    BufferedReader reader = Utils.getGZBufferedReader(input);
    PrintWriter writer = Utils.getPrintWriter(output);
    String line;
    int count = 0;
    while ((line = reader.readLine()) != null) {
      String[] row = line.split(",");
      for (int i = 0; i < row.length; i++) {
        if (i > 0) writer.print(",");
        if (!row[i].equals("0")) writer.print(row[i]);
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
}
