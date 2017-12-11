package com.lucasmarques.artificialintelligence.hopfield;

public class ActivationFunction {

    public static int stepFunction(double x) {
        if (x >= 0) {
            return 1;
        } else {
            return - 1;
        }
    }

}
