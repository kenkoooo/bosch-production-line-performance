import java.io.*;
import java.util.ArrayDeque;

public class ColumnReducer {
  private final String filename = "./resources/train_numeric.csv";
  private final String output = "./output/same_columns_numeric";
  private final long start = System.currentTimeMillis();

  private void run() throws IOException {
    int fullSize = Utils.getFullSize(filename);

    BufferedReader br = Utils.getBufferedReader(filename);
    String[] header = br.readLine().split(",");
    ArrayDeque<int[]> deque = new ArrayDeque<>();
    for (int i = 1; i < header.length; i++) {
      for (int j = i + 1; j < header.length; j++) {
        deque.add(new int[]{i, j});
      }
    }

    String line;
    String[] row;
    int count = 0;

    while ((line = br.readLine()) != null) {
      row = line.split(",");
      int size = deque.size();
      for (int i = 0; i < size; i++) {
        int[] ij = deque.poll();
        if (row.length > ij[0] && row.length <= ij[1]) continue;
        if (row.length <= ij[0] || row[ij[0]].equals(row[ij[1]])) deque.add(ij);
      }

      count++;
      if (count % 10000 == 0) {
        Utils.elapsedTime(start);
        System.out.println(((double) count * 100.0 / fullSize) + "%");
      }
    }

    PrintWriter writer = Utils.getPrintWriter(output);
    while (!deque.isEmpty()) {
      int[] ij = deque.poll();
      writer.println(header[ij[0]] + " " + header[ij[1]]);
    }
    Utils.elapsedTime(start);
  }

  public static void main(String[] args) throws IOException {
    ColumnReducer reducer = new ColumnReducer();
    reducer.run();
    Utils.elapsedTime(reducer.start);
  }
}
