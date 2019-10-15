package propagation

import (
	"go/neuronal/network/helper"

	"gonum.org/v1/gonum/mat"
)

func ForwardPropagation(xi *mat.Dense, thetaPerLayer []*mat.Dense) ([]*ForwardLayer, mat.Matrix) {
	fLayers := make([]*ForwardLayer, len(thetaPerLayer))
	ai := xi

	for i := 0; i < len(thetaPerLayer); i++ {
		ai_b := helper.AddBias(ai)
		Theta := thetaPerLayer[i]
		zi := helper.Multiply(ai_b, Theta)
		fLayers[i] = &ForwardLayer{
			AValue: helper.Copy(ai_b),
			Zvalue: helper.Copy(zi),
		}
		ai = helper.SigmoidMatrix(zi)
	}
	fLayers = append(fLayers, &ForwardLayer{
		AValue: ai,
	})
	return fLayers, ai

}

type ForwardLayer struct {
	AValue mat.Matrix
	Zvalue mat.Matrix
}
