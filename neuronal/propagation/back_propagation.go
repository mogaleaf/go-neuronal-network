package propagation

import (
	"go/neuronal/network/helper"

	"gonum.org/v1/gonum/mat"
)

func BackwardPropagation(yi int, flayers []*ForwardLayer, classNumber int, thetaPerLayer []*mat.Dense) []*BackwardLayer {
	returnLayer := make([]*BackwardLayer, len(flayers)-1)
	h := flayers[len(flayers)-1].AValue
	r, c := h.Dims()
	di := mat.NewDense(c, r, nil)
	for k := 0; k < classNumber; k++ {
		if (k + 1) == yi {
			di.Set(k, 0, h.At(0, k)-1.0)
		} else {
			di.Set(k, 0, h.At(0, k))
		}
	}

	for i := len(flayers) - 2; i >= 0; i-- {
		ti := thetaPerLayer[i]
		ti_m := helper.RemoveBias(ti)
		ai := flayers[i].AValue
		BDi := helper.Multiply(di, ai).T()
		returnLayer[i] = &BackwardLayer{
			BD: BDi,
		}
		if i > 0 {
			zi := helper.SigmoidGradMatrix(flayers[i-1].Zvalue.(*mat.Dense)).T()
			di = helper.Multiply(ti_m, di)
			di.Apply(func(i, j int, v float64) float64 {
				return v * zi.At(i, j)
			}, di)

		}
	}

	return returnLayer
}

type BackwardLayer struct {
	BD mat.Matrix
}
