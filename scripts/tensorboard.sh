#!/bin/bash

PORT="6006"
LOGDIR="runs"

if [ "$#" -ne 0 ]
then
	PORT="$1"
	shift
fi

if [ "$#" -ne 0 ]
then
	LOGDIR="$1"
	shift
fi

source virtualenvwrapper.sh
workon tensorboard
mkdir tmp
TMPDIR=./tmp/ tensorboard --logdir="$LOGDIR" --port="$PORT" $@
rm -rf tmp
deactivate

