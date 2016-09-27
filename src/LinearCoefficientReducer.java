import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Arrays;

public class LinearCoefficientReducer {
  class Q {
    int from, to;
    Double K = Double.NaN;
    double maxEPS = 0.0;
  }

  private final String filename = "./output/reduced_merged.csv.gz";
  private long start = System.currentTimeMillis();
  private double EPS = 1e-8;

  private void run() throws IOException {
    int fullSize = Utils.getFullSize(filename, true);
    BufferedReader reader = Utils.getGZBufferedReader(filename);
    String[] header = reader.readLine().split(",");

    int N = header.length;
    ArrayDeque<Q> deque = new ArrayDeque<>();
    for (int i = 0; i < N; i++) {
      for (int j = i + 1; j < N; j++) {
        Q q = new Q();
        q.from = i;
        q.to = j;
        deque.add(q);
      }
    }

    String line;
    String[] row;
    Double[] doubles = new Double[N];
    int count = 0;
    while ((line = reader.readLine()) != null) {
      Arrays.fill(doubles, Double.NaN);
      row = line.split(",");
      for (int i = 0; i < row.length; i++) {
        if (row[i].equals("")) continue;
        doubles[i] = Double.parseDouble(row[i]);
      }

      int s = deque.size();
      for (int i = 0; i < s; i++) {
        Q q = deque.pollFirst();
        if (doubles[q.from] == Double.NaN && doubles[q.to] == Double.NaN) {
          deque.addLast(q);
          continue;
        }
        if (doubles[q.from] == Double.NaN || doubles[q.to] == Double.NaN) continue;

        // 全ての値が同じ列は除いてあるはずなので, 0.0 が存在した場合、kを設定することはできない
        if (doubles[q.from] == 0.0 || doubles[q.to] == 0.0) continue;

        double k = doubles[q.from] / doubles[q.to];
        if (q.K == Double.NaN) {
          q.K = k;
          deque.addLast(q);
        } else if (Math.abs(q.K - k) < EPS) {
          q.maxEPS = Math.max(q.maxEPS, Math.abs(q.K - k));
          deque.addLast(q);
        }
      }

      count++;
      if (count % 10000 == 0) {
        Utils.elapsedTime(start);
        System.out.println(((double) count * 100.0 / fullSize) + "%");
      }
    }

    while (!deque.isEmpty()) {
      Q q = deque.pollFirst();
      System.out.print(header[q.from]);
      System.out.print(" ");
      System.out.print(header[q.to]);
      System.out.print(" ");
      System.out.print(q.K);
      System.out.print(" ");
      System.out.print(q.maxEPS);
      System.out.println();
    }
  }

  public static void main(String[] args) throws IOException {
    LinearCoefficientReducer reducer = new LinearCoefficientReducer();
    reducer.run();
    Utils.elapsedTime(reducer.start);
  }
}
