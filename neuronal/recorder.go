package neuronal

import (
	"fmt"

	"gonum.org/v1/gonum/optimize"
)

type neuronalRecorder struct {
	n *NeuronalNetwork
}

func (*neuronalRecorder) Init() error {
	return nil
}

func (n *neuronalRecorder) Record(l *optimize.Location, o optimize.Operation, s *optimize.Stats) error {
	if o == optimize.MajorIteration {
		//	println(fmt.Sprintf("Value: %0.02f", l.F))
	}

	return nil
}

func (n *neuronalRecorder) checkGrad(thetas []float64, calcGrad []float64) error {
	epsilon := 0.00010000
	numGrad := make([]float64, len(thetas))

	for i, _ := range numGrad {
		thetaPlusEp := make([]float64, 0)
		thetaPlusEp = append(thetaPlusEp, thetas...)
		thetaMinusEp := make([]float64, 0)
		thetaMinusEp = append(thetaMinusEp, thetas...)
		thetaPlusEp[i] += epsilon
		thetaMinusEp[i] -= epsilon
		loss2 := n.n.Cost(thetaMinusEp)
		loss1 := n.n.Cost(thetaPlusEp)
		numGrad[i] = (loss1 - loss2) / (2 * epsilon)
	}
	for i, _ := range numGrad {
		println(fmt.Sprintf("calculate: %0.9f vs epprox : %0.9f ", calcGrad[i], numGrad[i]))
	}
	return nil
}
