package neuronal

import (
	"encoding/csv"
	"math"
	"math/rand"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func InitThetaRandomValue(rowSize int, colSize int) *mat.Dense {
	theta := mat.NewDense(rowSize, colSize, nil)
	theta.Apply(func(i, j int, value float64) float64 {
		epsilon_init := 0.12
		return (rand.Float64() * 2 * epsilon_init) - epsilon_init
	}, theta)
	return theta
}

func LoadFileVectorized(fileName string) (*mat.Dense, *mat.Dense, error) {
	f, err := os.Open(fileName)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, nil, err
	}

	featureNumber := len(lines[0]) - 1
	m := (len(lines))
	X := mat.NewDense(m, featureNumber, nil)
	y := mat.NewDense(m, 1, nil)

	// Loop through lines & turn into object
	for i, line := range lines {

		for j, data := range line {
			f, err := strconv.ParseFloat(data, 64)
			if err != nil {
				return nil, nil, err
			}
			if j != len(line)-1 {
				X.Set(i, j, f)
				//X.Set(i, j+2*featureNumber, f*f*f)
			} else {
				y.Set(i, 0, f)
			}

		}

	}
	return X, y, nil
}

func NormalizeVectorized(X mat.Matrix) (*mat.Dense, *mat.Dense, *mat.Dense, error) {
	r, c := X.Dims()
	N := mat.NewDense(r, c, nil)
	S := mat.NewDense(1, c, nil)
	M := mat.NewDense(1, c, nil)

	for j := 0; j < c; j++ {
		var max float64
		min := math.MaxFloat64
		var sum float64
		for i := 0; i < r; i++ {
			if max < X.At(i, j) {
				max = X.At(i, j)
			}
			if min > X.At(i, j) {
				min = X.At(i, j)
			}
			sum += X.At(i, j)
		}
		mean := float64(sum / float64(r))
		M.Set(0, j, mean)
		s := max - min

		S.Set(0, j, s)

		for i := 0; i < r; i++ {

			N.Set(i, j, (X.At(i, j)-mean)/s)
		}
	}

	return N, M, S, nil
}
