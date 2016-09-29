import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;

public class MergedTestDataMaker {
  private final long start = System.currentTimeMillis();
  private String train = "./output/reduced_merged.csv.gz";
  private String test1 = "./resources/test_categorical.csv.gz";
  private String test2 = "./resources/test_date.csv.gz";
  private String test3 = "./resources/test_numeric.csv.gz";
  private String output = "./output/test_reduced_merged.csv";

  private void purify(String[] header, String[] header1) {
    for (int i = 0; i < header1.length; i++) {
      for (int j = 0; j < header.length; j++) {
        if (header[j].equals(header1[i])) break;
        if (j == header.length - 1) header1[i] = null;
      }
    }
  }

  private void run() throws IOException {
    int fullSize = Utils.getFullSize(test1, true);
    System.out.println(fullSize);
    Utils.elapsedTime(start);

    String[] header;
    {
      BufferedReader reader = Utils.getGZBufferedReader(train);
      header = reader.readLine().split(",");
    }
    BufferedReader reader1 = Utils.getGZBufferedReader(test1);
    BufferedReader reader2 = Utils.getGZBufferedReader(test2);
    BufferedReader reader3 = Utils.getGZBufferedReader(test3);

    String[] header1 = reader1.readLine().split(",");
    String[] header2 = reader2.readLine().split(",");
    String[] header3 = reader3.readLine().split(",");

    purify(header, header1);
    purify(header, header2);
    purify(header, header3);

    if (header2[0].equals(header1[0])) header2[0] = null;
    if (header3[0].equals(header1[0])) header3[0] = null;

    PrintWriter writer = Utils.getPrintWriter(output);
    for (int i = 0; i < header1.length; i++) {
      if (header1[i] == null) continue;
      if (i > 0) writer.print(",");
      writer.print(header1[i]);
    }
    for (String aHeader2 : header2) {
      if (aHeader2 == null) continue;
      writer.print(",");
      writer.print(aHeader2);
    }
    for (String aHeader3 : header3) {
      if (aHeader3 == null) continue;
      writer.print(",");
      writer.print(aHeader3);
    }
    writer.println();
    for (int count = 1; count < fullSize; count++) {
      String[] row1 = reader1.readLine().split(",");
      for (int i = 0; i < header1.length; i++) {
        if (header1[i] == null) continue;
        if (i > 0) writer.print(",");
        if (row1.length > i) writer.print(row1[i]);
      }
      String[] row2 = reader2.readLine().split(",");
      for (int i = 0; i < header2.length; i++) {
        if (header2[i] == null) continue;
        writer.print(",");
        if (row2.length > i) writer.print(row2[i]);
      }
      String[] row3 = reader3.readLine().split(",");
      for (int i = 0; i < header3.length; i++) {
        if (header3[i] == null) continue;
        writer.print(",");
        if (row3.length > i) writer.print(row3[i]);
      }
      writer.println();

      if (count % 10000 == 0) {
        Utils.elapsedTime(start);
        System.out.println(((double) count * 100.0 / fullSize) + "%");
      }
    }
    writer.close();
  }

  public static void main(String[] args) throws IOException {
    MergedTestDataMaker maker = new MergedTestDataMaker();
    maker.run();
    Utils.elapsedTime(maker.start);
  }
}
