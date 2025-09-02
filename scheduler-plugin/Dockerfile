# Multi-stage build for ML Scheduler Plugin
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git make bash

# Set working directory
WORKDIR /workspace

# Copy go modules files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY cmd/ cmd/
COPY pkg/ pkg/

# Build the scheduler plugin
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags='-w -s -extldflags "-static"' \
    -a -installsuffix cgo \
    -o ml-scheduler \
    ./cmd/main.go

# Final runtime image
FROM gcr.io/distroless/static:nonroot

# Copy the binary from builder
COPY --from=builder /workspace/ml-scheduler /usr/local/bin/kube-scheduler

# Use non-root user
USER 65532:65532

# Expose ports
EXPOSE 10259 10251

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD ["/usr/local/bin/kube-scheduler", "--version"]

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/kube-scheduler"]