package neuronal

import (
	"fmt"
	"go/neuronal/network/config"
	"go/neuronal/network/helper"
	"go/neuronal/network/neuronal/propagation"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

type NeuronalNetwork struct {
	LayersConfig config.LayersConfig
	layers       *Layers
	ClassNumber  int
	trainingData *TrainingData
	IterationMax int
	Lambda       float64
}

type TrainingData struct {
	input  *mat.Dense
	output *mat.Dense
}

func BuildNeuronalNetwork(config config.Config, X_Norm *mat.Dense, y *mat.Dense, layerconfig config.LayersConfig) *NeuronalNetwork {
	return &NeuronalNetwork{
		LayersConfig: layerconfig,
		ClassNumber:  config.ClassNumber,
		trainingData: &TrainingData{
			input:  X_Norm,
			output: y,
		},
		IterationMax: config.IterationNumber,
		Lambda:       config.Lambda,
	}

}

//Cost function of a neuronal network
func (n *NeuronalNetwork) Cost(thetas []float64) float64 {
	nlayers := n.unroll(thetas, n.layers)
	J := 0.0
	inputNumber, featuresNumbers := n.trainingData.input.Dims()
	for i := 0; i < inputNumber; i++ {
		rowDense := mat.NewDense(1, featuresNumbers, n.trainingData.input.RawRowView(i))
		_, Hi := propagation.ForwardPropagation(rowDense, nlayers)
		for k := 0; k < n.ClassNumber; k++ {
			if int(n.trainingData.output.At(i, 0)) == (k + 1) {
				if Hi.At(0, k) == 0 {
					J += 30
					break
				}
				J += -math.Log(Hi.At(0, k))
			} else {
				if Hi.At(0, k) == 1 {
					J += 30
					break
				}
				J += -math.Log(1.0 - Hi.At(0, k))
			}
		}
	}

	sumThetaSquare := 0.0
	for _, l := range nlayers {
		r, c := l.Dims()
		for a := 1; a < r; a++ {
			for b := 0; b < c; b++ {
				sumThetaSquare += l.At(a, b) * l.At(a, b)
			}
		}
	}

	reg := (n.Lambda * sumThetaSquare) / float64(2*inputNumber)
	cost := J*float64(1/float64(inputNumber)) + reg
	return cost
}

// Gradient of a neuronal network using forward and backward propagation
func (n *NeuronalNetwork) Grad(Grad, thetas []float64) {
	layers := n.unroll(thetas, n.layers)
	BD := make([]*mat.Dense, len(n.layers.Thetas))
	inputNumber, featuresNumbers := n.trainingData.input.Dims()
	for i := 0; i < inputNumber; i++ {
		rowDense := mat.NewDense(1, featuresNumbers, n.trainingData.input.RawRowView(i))
		forwardLayers, _ := propagation.ForwardPropagation(rowDense, layers)
		back := propagation.BackwardPropagation(int(n.trainingData.output.At(i, 0)), forwardLayers, n.ClassNumber, layers)

		for l := 0; l < len(back); l++ {
			if BD[l] == nil {
				r, c := back[l].BD.Dims()
				BD[l] = mat.NewDense(r, c, nil)
			}
			BD[l].Apply(func(a, b int, v float64) float64 {
				return v + back[l].BD.At(a, b)
			}, BD[l])
		}
	}

	for j := 0; j < len(BD); j++ {
		BD[j].Apply(func(a, b int, v float64) float64 {
			if a != 0 {
				return (1/float64(inputNumber))*v + (n.Lambda*layers[j].At(a, b))/float64(inputNumber)
			}
			return (1 / float64(inputNumber)) * v
		}, BD[j])
	}

	unroll := n.roll(BD)
	for i, t := range unroll {
		Grad[i] = t
	}

}

func (n *NeuronalNetwork) Train(X *mat.Dense, y *mat.Dense) (*Layers, error) {
	_, featuresNumbers := X.Dims()
	n.layers = NewLayers(n.LayersConfig, featuresNumbers)
	p := optimize.Problem{
		Func: n.Cost,
		Grad: n.Grad,
	}
	s := optimize.Settings{
		Recorder: &neuronalRecorder{
			n: n,
		},
		MajorIterations: n.IterationMax,
	}
	n.trainingData.input = X
	n.trainingData.output = y
	result, err := optimize.Minimize(p, n.layers.rollInit(), &s, &optimize.LBFGS{})
	if err != nil {
		return nil, err
	}
	unroll := n.unroll(result.X, n.layers)
	return &Layers{
		Thetas: unroll,
	}, nil
}

func (n *NeuronalNetwork) Accurate(X *mat.Dense, y *mat.Dense, layers *Layers) (float64, error) {
	n.trainingData.input = X
	n.trainingData.output = y
	n.layers = layers
	n.Lambda = 0
	cost := n.Cost(layers.rollInit())

	return cost, nil
}

func (n *NeuronalNetwork) Predict(X *mat.Dense, y *mat.Dense, layers *Layers) (float64, error) {
	n.trainingData.input = X
	n.trainingData.output = y
	_, featuresNumber := X.Dims()
	n.layers = layers

	n.Lambda = 0

	r, _ := X.Dims()
	nbTrue := 0
	for i := 0; i < r; i++ {
		yi := y.At(i, 0)
		rowDense := mat.NewDense(1, featuresNumber, X.RawRowView(i))
		hi := rowDense
		for _, t := range layers.Thetas {
			bias := helper.AddBias(hi)
			hi = helper.SigmoidMatrix(helper.Multiply(bias, t))
		}
		_, k := hi.Dims()
		max := -1 * math.MaxFloat64
		class := 0.0
		for ik := 0; ik < k; ik++ {
			if hi.At(0, ik) > max {
				max = hi.At(0, ik)
				class = float64(ik + 1)
			}
		}

		if class == yi {
			nbTrue++
		}

	}
	println(fmt.Sprintf("nbTrue %d", nbTrue))
	percentage := float64(nbTrue) / float64(r) * 100.0

	return percentage, nil
}

func (n *NeuronalNetwork) unroll(thetas []float64, layers *Layers) []*mat.Dense {
	index := 0
	newThetas := make([]*mat.Dense, 0)
	for _, l := range layers.Thetas {
		r, c := l.Dims()
		newThetas = append(newThetas, mat.NewDense(r, c, thetas[index:(r*c)+index]))
		index += r * c
	}
	return newThetas
}

func (n *NeuronalNetwork) roll(thetas []*mat.Dense) []float64 {
	var returnTab []float64
	for _, t := range thetas {
		r, c := t.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				returnTab = append(returnTab, t.At(i, j))
			}
		}
	}
	return returnTab
}
