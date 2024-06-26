.PHONY: build-all image-base image-cpp-linter publish-all publish-cpp-linter all publish-base base cpp-linter

build-all: image-base image-cpp-linter

publish-all: publish-base publish-cpp-linter

image-cpp-linter: image-base
	docker build -t pveleskopglc/chipStar:cpp-linter --progress=plain -f ./DockerfileCPPLinter .
image-base:
	docker build -t pveleskopglc/chipStar:base --progress=plain -f ./DockerfileBase .

publish-cpp-linter: image-cpp-linter
	docker push pveleskopglc/chipStar:cpp-linter
publish-base: image-base
	docker push pveleskopglc/chipStar:base

base: image-base publish-base

cpp-linter: image-cpp-linter publish-cpp-linter

all: base cpp-linter