package helper

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func SigmoidMatrix(mat *mat.Dense) *mat.Dense {
	mat.Apply(Sigmoid, mat)
	return mat
}

func Sigmoid(i, j int, value float64) float64 {
	return (1 / (1 + math.Exp(-value)))
}

func SigmoidGradMatrix(mat *mat.Dense) *mat.Dense {
	mat.Apply(SigmoidGrad, mat)
	return mat
}

func SigmoidGrad(i, j int, value float64) float64 {
	g := Sigmoid(i, j, value)
	return g * (1 - g)
}
