.PHONY: build-all image-base image-cpp-linter publish-all publish-cpp-linter all publish-base base cpp-linter

build-all: image-base image-cpp-linter

publish-all: publish-base publish-cpp-linter

image-cpp-linter: image-base
	docker build -t krisitanpn/teststar:cpp-linter --progress=plain -f ./DockerfileCPPLinter .
image-base:
	docker build -t krisitanpn/teststar:base --progress=plain -f ./DockerfileBase .

publish-cpp-linter: image-cpp-linter
	docker push krisitanpn/teststar:cpp-linter
publish-base: image-base
	docker push krisitanpn/teststar:base

base: image-base publish-base

cpp-linter: image-cpp-linter publish-cpp-linter

all: base cpp-linter