package main

import (
	"fmt"
	"go/neuronal/network/config"
	"go/neuronal/network/neuronal"
	"log"
	"math"

	"golang.org/x/image/colornames"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func main() {
	classNumber := 10
	X, y, error := neuronal.LoadFileVectorized("train.csv")
	if error != nil {
		panic(error)
	}
	X_Norm, M, S, error := neuronal.NormalizeVectorized(X)
	X.Dims()

	layersConfig := config.LayersConfig{
		HiddenLayers: []config.LayerConfig{
			{
				NodeNumber: 25,
			},
		},
		OutputLayer: config.LayerConfig{
			NodeNumber: classNumber,
		},
	}

	config := config.Config{
		IterationNumber: 50,
		Lambda:          0,
		ClassNumber:     classNumber,
	}
	network := neuronal.BuildNeuronalNetwork(config, X_Norm, y, layersConfig)

	layers, error := network.Train(X_Norm, y)
	// Load
	X_Test, y_Test, error := neuronal.LoadFileVectorized("test.csv")
	if error != nil {
		panic(error)
	}

	//apply normalization
	r, c := X_Test.Dims()
	X_Test_Norm := mat.NewDense(r, c, nil)
	X_Test_Norm.Copy(X_Test)
	X_Test_Norm.Apply(func(i, j int, v float64) float64 {
		return (v - M.At(0, j)) / S.At(0, j)
	}, X_Test_Norm)

	if error != nil {
		println(error.Error())
	}

	//1.728927022
	costTraining, error := network.Accurate(X_Norm, y, layers)
	if error != nil {
		println(error.Error())
	}
	println(fmt.Sprintf("Here is the accurate cost costTraining: %0.9f", costTraining))

	costCsv, error := network.Accurate(X_Test_Norm, y_Test, layers)
	if error != nil {
		println(error.Error())
	}
	println(fmt.Sprintf("Here is the accurate cost csv: %0.9f", costCsv))

	predict, error := network.Predict(X_Norm, y, layers)
	println(fmt.Sprintf("Here is the predict train percentage x: %0.9f", predict))

	predictNew, error := network.Predict(X_Test_Norm, y_Test, layers)
	println(fmt.Sprintf("Here is the predict test percentage test: %0.9f", predictNew))
	//showCostM(network,featuresNumber,X_Norm,y,X_Test_Norm,y_Test,m)
	showPolynomial(network, X, y, X_Test, y_Test, 8)

}

func printPredict(network *neuronal.NeuronalNetwork, X_Norm *mat.Dense, y *mat.Dense, X_Test *mat.Dense, y_Test *mat.Dense, layers *neuronal.Layers) {
	predict, _ := network.Predict(X_Norm, y, layers)
	println(fmt.Sprintf("Here is the predict train percentage x: %0.9f", predict))

	predictNew, _ := network.Predict(X_Test, y_Test, layers)
	println(fmt.Sprintf("Here is the predict test percentage test: %0.9f", predictNew))
}

func showPolynomial(network *neuronal.NeuronalNetwork, X *mat.Dense, y *mat.Dense, X_Test *mat.Dense, y_Test *mat.Dense, d int) {

	pts1 := make(plotter.XYs, 0)
	pts2 := make(plotter.XYs, 0)
	for i := 1; i < d; i++ {
		println(fmt.Sprintf(" iter %d", i))
		X_poly := addPoly(X, i)
		X_Poly_Norm, M, S, error := neuronal.NormalizeVectorized(X_poly)

		newLayers, error := network.Train(X_Poly_Norm, y)
		costI, error := network.Accurate(X_Poly_Norm, y, newLayers)
		if error != nil {
			println(error.Error())
		}
		pts1 = append(pts1, plotter.XY{
			X: float64(i),
			Y: costI,
		})

		X_Test_Poly := addPoly(X_Test, i)
		r, c := X_Test_Poly.Dims()
		X_Test_Poly_Norm := mat.NewDense(r, c, nil)
		X_Test_Poly_Norm.Copy(X_Test_Poly)
		X_Test_Poly_Norm.Apply(func(i, j int, v float64) float64 {
			return (v - M.At(0, j)) / S.At(0, j)
		}, X_Test_Poly_Norm)

		costCsvM, error := network.Accurate(X_Test_Poly_Norm, y_Test, newLayers)
		pts2 = append(pts2, plotter.XY{
			X: float64(i),
			Y: costCsvM,
		})
		println(fmt.Sprintf("print i = %d", i))
		show(pts1, pts2, i, "cost_d", "d")

		printPredict(network, X_Poly_Norm, y, X_Test_Poly_Norm, y_Test, newLayers)

	}
	show(pts1, pts2, d, "cost_d", "d")
}

func addPoly(X *mat.Dense, poly int) *mat.Dense {
	r, c := X.Dims()
	slice := mat.NewDense(r, (poly * c), nil)
	for a := 0; a < r; a++ {
		for b := 0; b < c; b++ {
			slice.Set(a, b, X.At(a, b))
			for k := 1; k < poly; k++ {
				slice.Set(a, k*c+b, math.Pow(X.At(a, b), float64(k+1)))
			}
		}
	}
	return slice
}

func showCostM(network *neuronal.NeuronalNetwork, featuresNumber int, X_Norm *mat.Dense, y *mat.Dense, X_Test *mat.Dense, y_Test *mat.Dense, m int) {
	pts1 := make(plotter.XYs, 0)
	pts2 := make(plotter.XYs, 0)
	for i := 0; i < m; i++ {
		slice := X_Norm.Slice(0, i+1, 0, featuresNumber)
		yslice := y.Slice(0, i+1, 0, 1)
		newLayers, error := network.Train(slice.(*mat.Dense), yslice.(*mat.Dense))
		costI, error := network.Accurate(slice.(*mat.Dense), yslice.(*mat.Dense), newLayers)
		if error != nil {
			println(error.Error())
		}
		pts1 = append(pts1, plotter.XY{
			X: float64(i),
			Y: costI,
		})
		costCsvM, error := network.Accurate(X_Test, y_Test, newLayers)
		pts2 = append(pts2, plotter.XY{
			X: float64(i),
			Y: costCsvM,
		})
		if i != 0 && i%10 == 0 {
			println(fmt.Sprintf("print i = %d", i))
			show(pts1, pts2, i, "cost_m", "m")
		}

	}
	show(pts1, pts2, m, "cost_m", "m")
}

func show(pts1 plotter.XYs, pts2 plotter.XYs, i int, name string, axis string) {
	p, err := plot.New()
	if err != nil {
		log.Panic(err)
	}
	p.Title.Text = "Data point"
	p.Y.Label.Text = "cost"
	p.X.Label.Text = axis
	p.Add(plotter.NewGrid())

	line1, pp1, err := plotter.NewLinePoints(pts1)
	if err != nil {
		log.Panic(err)
	}
	line1.Color = colornames.Red
	pp1.Shape = draw.PlusGlyph{}
	p.Add(line1)
	p.Add(pp1)

	line2, pp2, err := plotter.NewLinePoints(pts2)
	line2.Color = colornames.Aqua
	pp2.Shape = draw.PlusGlyph{}
	p.Add(line2)
	p.Add(pp2)

	err = p.Save(10*vg.Centimeter, 5*vg.Centimeter, fmt.Sprintf("%s_%d.png", name, i))
	if err != nil {
		log.Panic(err)
	}
}
