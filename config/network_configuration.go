package config

type Config struct {
	Lambda          float64
	IterationNumber int
	ClassNumber     int
}

type InputLayerConfig struct {
	InputNumber int
}

type LayerConfig struct {
	NodeNumber int
}

type LayersConfig struct {
	HiddenLayers     []LayerConfig
	OutputLayer      LayerConfig
	TrainingFilePath string
}
