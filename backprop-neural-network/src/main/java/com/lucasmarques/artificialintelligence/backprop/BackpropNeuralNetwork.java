package com.lucasmarques.artificialintelligence.backprop;

public class BackpropNeuralNetwork {

    private Layer[] layers;

    public BackpropNeuralNetwork(int inputSize, int hiddenLayer, int outputSize) {
        this.layers = new Layer[2];
        this.layers[0] = new Layer(inputSize, hiddenLayer);
        this.layers[1] = new Layer(hiddenLayer, outputSize);
    }

    public Layer getLayer(int index) {
        return this.layers[index];
    }

    public float[] run(float[] input) {

        float[] activations = input;

        for (int i = 0; i < this.layers.length; i++) {
            activations = layers[i].run(activations);
        }

        return activations;

    }

    public void train(float[] input, float[] targetOutput, float learningRate, float momentum) {

        float[] calculatedOutput = run(input);
        float[] error = new float[calculatedOutput.length];

        for (int i = 0; i < error.length; i++) {
            error[i] = targetOutput[i] - calculatedOutput[i];
        }

        for (int i = this.layers.length - 1; i >= 0; i--) {
            error = this.layers[i].train(error, learningRate, momentum);
        }

    }

}
