DOCKER=docker
IMGTAG=alexdarancio7/stelar_field_segmentation:latest
IMGPATH=.
DOCKERFILE=$(IMGPATH)/Dockerfile

.PHONY: all build push

all: build push

build:
	$(DOCKER) build -f $(DOCKERFILE) $(IMGPATH) -t $(IMGTAG)

push:
	$(DOCKER) push $(IMGTAG)

