import org.encog.engine.network.activation.ActivationElliott;
import org.encog.engine.network.activation.ActivationStep;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import java.util.ArrayList;

public class NeuralNet {

    private static final double TRAIN_CONVERGENCE_THRESHOLD = 0.005;
    private static final int MAX_EPOCHS = 1000;
    private ArrayList<double[][]> valid;
    private ArrayList<double[][]> invalid;
    private BasicNetwork network;
    private int inputLayerCount;

    public NeuralNet() {
        this.inputLayerCount = -1;
        this.network = new BasicNetwork();
    }

    /**
     * Sets the array size of the data sent to the network to train. Must be called before the
     * network is actually trained and used
     * @param inputLayerCount The value
     */
    public void setInputLayerCount(int inputLayerCount) {
        this.inputLayerCount = inputLayerCount;
        valid = new ArrayList<>();
        invalid = new ArrayList<>();
        network.addLayer(new BasicLayer(new ActivationElliott(), false, inputLayerCount)); // Input
        network.addLayer(new BasicLayer(new ActivationElliott(), false, inputLayerCount)); // Hidden
        network.addLayer(new BasicLayer(new ActivationStep(), false, 1)); // Output
        network.getStructure().finalizeStructure();
        network.reset();
    }

    /**
     * Adds an entry to the System, and if indicated, retrain the network.
     * @param data     The data as input
     * @param validity If the data was valid, the output
     * @param retrain  If the network should be retrained
     */
    public void addEntry(double[][] data, boolean validity, boolean retrain) {
        ArrayList<double[][]> list = validity ? valid : invalid;
        list.add(data);
        if (retrain) train();
    }

    /**
     * Uses the given coordinate data to predict if they are valid or not
     * @param coordinates The coordinate data
     */
    public boolean predict(double[] coordinates) {
        MLData input = new BasicMLData(coordinates);
        MLData output = network.compute(input);
        return output.getData()[0] == 1;
    }

    public void train() {
        double[][] validArray = new double[valid.size()][inputLayerCount];
        double[][] invalidArray = new double[invalid.size()][inputLayerCount];

        for (int i = 0; i < valid.size(); i++) validArray = valid.get(i);
        for (int i = 0; i < invalid.size(); i++) invalidArray = invalid.get(i);

        train(validArray, invalidArray);
    }

    /**
     * Takes in data to train the network.
     *
     * @param validCoordinates   Inputs of coordinates that would be valid.
     * @param invalidCoordinates Like the first parameter, but with invalid data
     */
    public void train(double[][] validCoordinates, double[][] invalidCoordinates) {
        // Creates input and output arrays
        int total = validCoordinates.length + invalidCoordinates.length;
        double[][] output = new double[total][1];
        double[][] input = new double[total][inputLayerCount];
        for (int i = 0; i < validCoordinates.length; i++) {
            input[i] = validCoordinates[i];
            output[i][0] = 1;
        }
        for (int i = 0; i < invalidCoordinates.length; i++) {
            input[i + validCoordinates.length] = invalidCoordinates[i];
            output[i + validCoordinates.length][0] = 0;
        }

        MLDataSet trainingSet = new BasicNeuralDataSet(input, output);
        final Train train = new ResilientPropagation(network, trainingSet);
        int counter = 0;
        double error;
        do {
            train.iteration();
            counter++;
            error = train.getError();
            //if (counter % 10 == 0) System.out.println(error);
        } while (error > TRAIN_CONVERGENCE_THRESHOLD && counter < MAX_EPOCHS);
    }
}
