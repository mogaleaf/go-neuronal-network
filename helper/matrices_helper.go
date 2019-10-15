package helper

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

func AddBias(matToAugment mat.Matrix) *mat.Dense {
	r, c := matToAugment.Dims()
	dataOnes := make([]float64, r)
	for i := range dataOnes {
		dataOnes[i] = 1
	}
	onesMatrice := mat.NewDense(r, 1, dataOnes)
	returnValue := mat.NewDense(r, c+1, nil)
	returnValue.Augment(onesMatrice, matToAugment)
	return returnValue
}

func RemoveBias(matToReduce mat.Matrix) *mat.Dense {
	r, c := matToReduce.Dims()
	return matToReduce.(*mat.Dense).Slice(1, r, 0, c).(*mat.Dense)
}

func Multiply(mat1 mat.Matrix, mat2 mat.Matrix) *mat.Dense {
	r, _ := mat1.Dims()
	_, c := mat2.Dims()
	returnValue := mat.NewDense(r, c, nil)
	returnValue.Mul(mat1, mat2)
	return returnValue
}

func Add(mat1 mat.Matrix, mat2 mat.Matrix) *mat.Dense {
	r, _ := mat1.Dims()
	_, c := mat2.Dims()
	returnValue := mat.NewDense(r, c, nil)
	returnValue.Add(mat1, mat2)
	return returnValue
}

func PrintMatrix(mat1 mat.Matrix) {
	r, c := mat1.Dims()
	for i := 0; i < r; i++ {
		println()
		for j := 0; j < c; j++ {
			print(fmt.Sprintf(" %0.5f", mat1.At(i, j)))
		}
	}
	println()
}

func Copy(matR *mat.Dense) *mat.Dense {
	r, c := matR.Dims()
	returnValue := mat.NewDense(r, c, nil)
	returnValue.Copy(matR)
	return returnValue
}
