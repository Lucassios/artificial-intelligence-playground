package com.lucasmarques.artificialintelligence.backprop;

public class ActivationFunction {

    public static float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }

    public static float dSigmoid(float sigmoid) {
//        return sigmoid(x) * (1 - sigmoid(x));
        return sigmoid * (1 - sigmoid);
    }

}
