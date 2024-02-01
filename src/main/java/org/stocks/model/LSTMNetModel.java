package org.stocks.model;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by zhanghao on 26/7/17.
 * @author ZHANG HAO
 */
public class LSTMNetModel {
	
    private static final int seed = 12345;

    private static final int lstmLayer1Size = 256;
    private static final int lstmLayer2Size = 256;
    private static final int denseLayerSize = 32;
    private static final double dropoutRatio = 0.2;
    private static final int truncatedBPTTLength = 22;

    public static MultiLayerNetwork buildLstmNetworks(int nIn, int nOut) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(nIn)
                        .nOut(lstmLayer1Size)
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.HARDSIGMOID)
                        .dropOut(dropoutRatio)
                        .build())
                .layer(1, new LSTM.Builder()
                        .nIn(lstmLayer1Size)
                        .nOut(lstmLayer2Size)
                        .activation(Activation.TANH)
                        .gateActivationFunction(Activation.HARDSIGMOID)
                        .dropOut(dropoutRatio)
                        .build())
                .layer(2, new DenseLayer.Builder()
                		.nIn(lstmLayer2Size)
                		.nOut(denseLayerSize)
                		.activation(Activation.RELU)
                		.build())
                .layer(3, new RnnOutputLayer.Builder()
                        .nIn(denseLayerSize)
                        .nOut(nOut)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(truncatedBPTTLength)
                .tBPTTBackwardLength(truncatedBPTTLength)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        return net;
    }
}