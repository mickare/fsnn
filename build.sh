#!/bin/sh
echo "Building..."

DIR="$(dirname "$(realpath "$0")")"

exit_on_error () {
	echo "An error occured! $1"
	exit 1
}

BIN="bin"
RUNNABLE="predictors"

CLEAN=0
RUN=0

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
        "clean") CLEAN=1
            ;;
   		"run") RUN=1
            ;;   		
    esac
    shift
done

# Go to script location
cd $DIR || exit_on_error "Failed to enter script location";

# Clean
if [ $CLEAN = 1 ]; then
	rm -rf "$BIN" && echo "$BIN folder removed."
fi

# Create bin folder
if [ ! -d "$BIN" ]; then
	mkdir -p "$BIN" && echo "$BIN folder created." || exit_on_error "Failed to create $BIN folder!"
fi

cd "$BIN" || exit_on_error "Failed to enter $BIN folder!"

# Stop if error in cmake or make
set -e

if [ ! -f "Makefile" ]; then
	cmake -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ ..
fi

make

# Dont stop anymore
set +e

if [ $RUN = 1 ]; then
	[ ! -f "$RUNNABLE" ] &&	exit_on_error "Runnable '$RUNNABLE' not found!"
	echo "Executing '$RUNNABLE'..."
	#.$RUNABLE
	exec "$DIR/$BIN/$RUNNABLE"
fi