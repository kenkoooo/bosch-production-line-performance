import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Arrays;

public class BrickColumnReducer {
  private final String gzFilename = "./output/reduced_merged_removeT.csv.gz";
  private long start = System.currentTimeMillis();
  private void run() throws IOException {
    BufferedReader reader = Utils.getGZBufferedReader(gzFilename);
    String[] header = reader.readLine().split(",");
    int N = header.length;

    ArrayDeque<Integer> dead = new ArrayDeque<>();
    for (int i = 0; i < N; i++) dead.add(i);

    int fullSize = Utils.getFullSize(gzFilename, true);
    String line;
    String[] prev = Arrays.copyOf(reader.readLine().split(","), N);
    int count = 0;
    while ((line = reader.readLine()) != null) {
      String[] row = Arrays.copyOf(line.split(","), N);
      int s = dead.size();
      for (int i = 0; i < s; i++) {
        int pos = dead.poll();
        if (row.length <= pos || prev.length <= pos) continue;
        if (row[pos].equals(prev[pos])) {
          dead.add(pos);
        }
      }

      prev = row;
      count++;
      if (count % 10000 == 0) {
        Utils.elapsedTime(start);
        System.out.println(((double) count * 100.0 / fullSize) + "%");
      }
    }

    while (!dead.isEmpty()) {
      System.out.println(header[dead.poll()]);
    }
  }

  public static void main(String[] args) throws IOException {
    BrickColumnReducer reducer = new BrickColumnReducer();
    reducer.run();
    Utils.elapsedTime(reducer.start);
  }
}
