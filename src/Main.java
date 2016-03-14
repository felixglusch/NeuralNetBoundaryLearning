public class Main {
    private static NeuralNet learner;
    private static final int BOUND_MAX = 10;
    public static void main(String[] args) throws InterruptedException {
        for (int k = 1; k <= 1; k++) {
            learner = new NeuralNet();
            double[][] valid = new double[10000][2];
            for (int i = 0; i < 10000; i++) {
                double sampleX = Math.round(Math.random() * BOUND_MAX);
                double sampleY = Math.round(Math.random() * BOUND_MAX);
                valid[i][0] = sampleX;
                valid[i][1] = sampleY;
            }
            double[][] invalid = new double[10000][2];
            for (int i = 0; i < 10000; i++) {
                double sampleX = Math.round(Math.random() * BOUND_MAX) + 5;
                double sampleY = Math.round(Math.random() * BOUND_MAX) + 5;
                invalid[i][0] = sampleX;
                invalid[i][1] = sampleY;
            }
            learner.setInputLayerCount(2);
            System.out.print("Before training: ");
            test(10000);
            learner.addEntry(valid, true, false);
            learner.addEntry(invalid, false, false);
            learner.train(valid, invalid);
            System.out.print("After training: ");
            test(10000);
        }
    }

    private static void test(int coordinates) {
        double violations = 0.0;
        for (int i = 0; i < coordinates; i++) {
            double x = Math.round(Math.random() * 15);
            double y = Math.round(Math.random() * 15);
            boolean guess = learner.predict(new double[]{x, y});
            boolean outside = x > BOUND_MAX || y > BOUND_MAX;
            if ((outside && guess) || (!outside && !guess))
                violations += 1;
        }
        System.out.println(1 - violations / coordinates);
    }
}
