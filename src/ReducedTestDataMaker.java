import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.TreeSet;

public class ReducedTestDataMaker {
  private final long start = System.currentTimeMillis();
  private String train = "./output/reduced_train_date.csv.gz";
  private String test = "./resources/test_date.csv.gz";
  private String output = "./output/reduced_test_date.csv";

  private void purify(String[] trainHeader, String[] testHeader) {
    TreeSet<String> trainSet = new TreeSet<>();
    Collections.addAll(trainSet, trainHeader);
    for (int i = 0; i < testHeader.length; i++) {
      if (!trainSet.contains(testHeader[i])) testHeader[i] = null;
    }
  }

  private void run() throws IOException {
    String[] trainHeader = Utils.getGZBufferedReader(train).readLine().split(",");
    BufferedReader reader = Utils.getGZBufferedReader(test);
    String[] testHeader = reader.readLine().split(",");
    purify(trainHeader, testHeader);

    PrintWriter writer = Utils.getPrintWriter(output);
    write(writer, testHeader, testHeader);
    String line;
    int count = 0;
    while ((line = reader.readLine()) != null) {
      String[] row = line.split(",");
      write(writer, testHeader, row);

      count++;
      if (count % 10000 == 0) {
        Utils.elapsedTime(start);
        System.out.println(count);
      }
    }
    writer.close();
  }

  private void write(PrintWriter writer, String[] header, String[] row) {
    for (int i = 0; i < header.length; i++) {
      if (header[i] == null) continue;
      if (i > 0) writer.print(",");
      if (row.length > i) writer.print(row[i]);
      writer.println();
    }
  }

  public static void main(String[] args) throws IOException {
    ReducedTestDataMaker maker = new ReducedTestDataMaker();
    maker.run();
    Utils.elapsedTime(maker.start);
  }
}
