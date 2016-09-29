import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayDeque;

public class ColumnReducer {
  private final long start = System.currentTimeMillis();
  private String input = "./output/reduced_train_categorical_binary.csv.gz";
  private String output = input + "_reduced.csv";

  private void run() throws IOException {
    BufferedReader reader = Utils.getGZBufferedReader(input);

    ArrayDeque<int[]> pairs = new ArrayDeque<>();
    ArrayDeque<Integer> brick = new ArrayDeque<>();
    String[] header = reader.readLine().split(",");
    int N = header.length;
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) pairs.add(new int[]{i, j});
    for (int i = 0; i < N; i++) brick.add(i);

    String line;
    String[] prev = null;
    int count = 0;
    while ((line = reader.readLine()) != null) {
      String[] row = line.split(",");
      int s = pairs.size();
      for (int i = 0; i < s; i++) {
        int[] pair = pairs.pollFirst();
        if (pair[0] >= row.length) {
          pairs.addLast(pair);
        } else if (pair[1] < row.length) {
          if (row[pair[0]].equals(row[pair[1]])) pairs.addLast(pair);
        }
      }
      if (prev == null) {
        prev = row;
        continue;
      }

      int t = brick.size();
      for (int i = 0; i < t; i++) {
        int b = brick.pollFirst();
        if (row.length <= b && prev.length <= b) brick.addLast(b);
        else if (row.length > b && prev.length > b && row[b].equals(prev[b])) brick.addLast(b);
      }
      prev = row;

      count++;
      if (count % 10000 == 0) {
        Utils.elapsedTime(start);
        System.out.println(count);
      }
    }

    while (!pairs.isEmpty()) {
      int[] pair = pairs.pollFirst();
      if (header[pair[1]] != null) System.out.println(header[pair[1]] + " is duplicated.");
      header[pair[1]] = null;
    }
    while (!brick.isEmpty()) {
      int b = brick.pollFirst();
      if (header[b] != null) System.out.println(header[b] + " is brick.");
      header[b] = null;
    }

    reader = Utils.getGZBufferedReader(input);
    PrintWriter writer = Utils.getPrintWriter(output);

    for (int i = 0; i < N; i++) {
      if (header[i] == null) continue;
      if (i > 0) writer.print(",");
      writer.print(header[i]);
    }
    writer.println();
    reader.readLine();

    count = 0;
    while ((line = reader.readLine()) != null) {
      String[] row = line.split(",");
      for (int i = 0; i < N; i++) {
        if (header[i] == null) continue;
        if (i > 0) writer.print(",");
        if (row.length > i) writer.print(row[i]);
      }

      count++;
      if (count % 10000 == 0) {
        Utils.elapsedTime(start);
        System.out.println(count);
      }
    }
    writer.close();
  }

  public static void main(String[] args) throws IOException {
    ColumnReducer maker = new ColumnReducer();
    maker.run();
    Utils.elapsedTime(maker.start);
  }
}
