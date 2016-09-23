import java.io.*;
import java.util.ArrayList;
import java.util.TreeSet;

public class ReducedCSVMaker {
  private final String input = "./output/reduced_merged_removeT.csv.gz";
  private final String output = "./output/reduced_merged.csv";
  private final String columns = "./output/same_value_columns";
  final long start = System.currentTimeMillis();

  private void run() throws IOException {
    TreeSet<String> names = deadColumnNames();
    BufferedReader br = Utils.getGZBufferedReader(input);
    String[] header = br.readLine().split(",");
    boolean[] dead = new boolean[header.length];
    for (int i = 0; i < header.length; i++) {
      if (names.contains(header[i])) dead[i] = true;
    }
    ArrayList<Integer> list = new ArrayList<>();
    for (int i = 0; i < header.length; i++) {
      if (!dead[i]) list.add(i);
    }

    PrintWriter pw = Utils.getPrintWriter(output);

    // Write header
    for (int i : list) {
      if (i > 0) pw.print(",");
      pw.print(header[i]);
    }
    pw.println();
    System.out.println(list.size() + " columns");

    // Read & Write
    int fullSize = Utils.getFullSize(input, true);
    int count = 0;
    String line;
    String[] row;
    while ((line = br.readLine()) != null) {
      row = line.split(",");
      for (int i : list) {
        if (i > 0) pw.print(",");
        if (row.length > i) pw.print(row[i]);
      }
      pw.println();

      count++;
      if (count % 10000 == 0) {
        Utils.elapsedTime(start);
        System.out.println(((double) count * 100.0 / fullSize) + "%");
      }

    }
    pw.close();
  }

  private TreeSet<String> deadColumnNames() throws IOException {
    TreeSet<String> dead = new TreeSet<>();

    BufferedReader reader = Utils.getBufferedReader(columns);
    String line;
    while ((line = reader.readLine()) != null) {
      dead.add(line);
    }
    return dead;
  }

  public static void main(String[] args) throws IOException {
    new ReducedCSVMaker().run();
  }

}
