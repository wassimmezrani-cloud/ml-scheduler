package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"k8s.io/component-base/cli"
	"k8s.io/component-base/logs"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"

	"github.com/hydatis/ml-scheduler-plugin/pkg/framework"
)

func main() {
	command := app.NewSchedulerCommand(
		app.WithPlugin(framework.Name, framework.New),
	)

	logs.InitLogs()
	defer logs.FlushLogs()

	ctx := context.Background()
	if err := command.ExecuteContext(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func init() {
	// Add custom flags
	flag.StringVar(&framework.PluginConfig.MLServicesURL, "ml-services-url", 
		"http://combined-ml-scorer:8080", "URL for ML services endpoint")
	flag.StringVar(&framework.PluginConfig.RedisURL, "redis-url", 
		"redis://redis-cache-service:6379", "Redis cache URL")
	flag.IntVar(&framework.PluginConfig.ScoringTimeoutMs, "scoring-timeout-ms", 
		5000, "ML scoring timeout in milliseconds")
	flag.BoolVar(&framework.PluginConfig.EnableFallback, "enable-fallback", 
		true, "Enable fallback to default scheduler")
	flag.StringVar(&framework.PluginConfig.LogLevel, "log-level", 
		"info", "Log level (debug, info, warn, error)")
	flag.Float64Var(&framework.PluginConfig.ConfidenceThreshold, "confidence-threshold", 
		0.7, "Minimum confidence threshold for ML recommendations")
}