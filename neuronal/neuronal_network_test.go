package neuronal

import (
	"go/neuronal/network/helper"
	"testing"

	"gonum.org/v1/gonum/mat"
)

//input_layer_size = 3;
//hidden_layer_size = 5;
//num_labels = 3;
//m = 5;

func TestGetOrganizationGroup(t *testing.T) {
	layers := make([]*Layer, 2)
	layers[0] = &Layer{
		Theta: mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
	}
	layers[1] = &Layer{
		Theta: mat.NewDense(3, 2, []float64{1, 1, 1, 1, 1, 1}),
	}
	network := &NeuronalNetwork{
		layers:      layers,
		ClassNumber: 2,
		IsTraining:  true,
		trainingData: &TrainingData{
			input:  mat.NewDense(3, 1, []float64{1, 2, 3}),
			output: mat.NewDense(3, 1, []float64{1, 2, 2}),
		},
		InputNumber:     3,
		FeaturesNumbers: 1,
	}
	//cost := network.Cost(network.rollInit())
	//println(fmt.Sprintf("%0.9f",cost))
	network.Grad(network.rollInit(), network.rollInit())

	//0.14893   0.22127
	//0.14893   0.22127

	//1.83959   1.78460   1.78460
	//0.83959   0.73081   0.73081

	//cost =  2.9869 986851712
	//0.049643   0.049643
	//0.049643   0.049643
	//0.073758   0.073758
	//0.073758   0.073758
	//0.613196   0.613196
	//0.279862   0.279862
	//0.594868   0.594868
	//0.243604   0.243604
	//0.594868   0.594868
	//0.243604   0.243604
}
