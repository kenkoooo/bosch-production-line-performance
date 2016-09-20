import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.TreeSet;

public class DataMerger {
  private final String categorical = "./output/reduced_categorical.csv";
  private final String date = "./output/reduced_date.csv";
  private final String numeric = "./resources/train_numeric.csv";

  private final String output = "./output/reduced_merged.csv";

  private final long start = System.currentTimeMillis();

  public static void main(String[] args) throws IOException {
    DataMerger merger = new DataMerger();
    merger.run();
    Utils.elapsedTime(merger.start);
  }

  private void countColumns() throws IOException {
    BufferedReader c = Utils.getBufferedReader(categorical);
    BufferedReader d = Utils.getBufferedReader(date);
    BufferedReader n = Utils.getBufferedReader(numeric);
    TreeSet<String> set = new TreeSet<>();
    int size = 0;
    System.out.println(set.size() - size);
    size = set.size();
    Collections.addAll(set, c.readLine().split(","));
    System.out.println(set.size() - size);
    size = set.size();
    Collections.addAll(set, d.readLine().split(","));
    System.out.println(set.size() - size);
    size = set.size();
    Collections.addAll(set, n.readLine().split(","));
    System.out.println(set.size() - size);
    System.out.println(set.size());
  }

  private void run() throws IOException {
//    if (!validate()) return;
//    System.out.println("All files are validated.");

    int fullSize = Utils.getFullSize(date);

    BufferedReader c = Utils.getBufferedReader(categorical);
    BufferedReader d = Utils.getBufferedReader(date);
    BufferedReader n = Utils.getBufferedReader(numeric);

    PrintWriter writer = Utils.getPrintWriter(output);
    int count = 0;
    String line;
    while ((line = c.readLine()) != null) {
      StringBuilder db = new StringBuilder(d.readLine());
      while (db.charAt(0) != ',') db.deleteCharAt(0);
      StringBuilder nb = new StringBuilder(n.readLine());
      while (nb.charAt(0) != ',') nb.deleteCharAt(0);

      writer.print(line);
      writer.print(db.toString());
      writer.print(nb.toString());
      writer.println();

      count++;
      if (count % 10000 == 0) {
        Utils.elapsedTime(start);
        System.out.println(((double) count * 100.0 / fullSize) + "%");
      }
    }
    writer.close();
  }

  /**
   * Id の並びが全ファイル一緒であることを確かめる
   *
   * @return
   * @throws IOException
   */
  private boolean validate() throws IOException {
    TreeSet<Integer> set = new TreeSet<>();
    if (!check(categorical, set)) return false;
    System.out.println("categorical ok");
    int size = set.size();
    if (!check(date, set)) return false;
    System.out.println("date ok");
    if (size != set.size()) return false;
    if (!check(numeric, set)) return false;
    System.out.println("numeric ok");
    return size == set.size();
  }

  /**
   * Id が全て昇順かチェックする
   *
   * @param filename
   * @param set
   * @return
   * @throws IOException
   */
  private boolean check(String filename, TreeSet<Integer> set) throws IOException {
    BufferedReader reader = Utils.getBufferedReader(filename);
    if (!reader.readLine().split(",")[0].equals("Id")) return false;
    int cur = 0;
    String line;
    while ((line = reader.readLine()) != null) {
      int next = Integer.parseInt(line.split(",")[0]);
      if (cur >= next) return false;
      cur = next;
      set.add(next);
    }
    return true;
  }
}
