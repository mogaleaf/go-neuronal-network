package neuronal

import (
	"go/neuronal/network/config"

	"gonum.org/v1/gonum/mat"
)

func (n *Layers) rollInit() []float64 {
	var returnTab []float64
	for _, t := range n.Thetas {
		r, c := t.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				returnTab = append(returnTab, t.At(i, j))
			}
		}
	}
	return returnTab
}

type Layers struct {
	Thetas []*mat.Dense
}

func NewLayers(config config.LayersConfig, featuresNumber int) *Layers {
	thetas := make([]*mat.Dense, len(config.HiddenLayers)+1)
	var thetaCol int
	if len(config.HiddenLayers) > 0 {
		thetaCol = config.HiddenLayers[0].NodeNumber
	} else {
		thetaCol = config.OutputLayer.NodeNumber
	}
	thetas[0] = InitThetaRandomValue(featuresNumber+1, thetaCol)

	for i := 0; i < len(config.HiddenLayers); i++ {
		currentNodeNumber := config.HiddenLayers[i].NodeNumber
		nextNodeNumber := 0
		if i == len(config.HiddenLayers)-1 {
			nextNodeNumber = config.OutputLayer.NodeNumber

		} else {
			nextNodeNumber = config.HiddenLayers[i+1].NodeNumber
		}
		thetas[i+1] = InitThetaRandomValue(currentNodeNumber+1, nextNodeNumber)
	}

	return &Layers{
		Thetas: thetas,
	}
}
