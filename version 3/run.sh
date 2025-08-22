while getopts "p:" opt; do
      case $opt in
        p) PROCESSES="$OPTARG" ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
      esac
    done

./build/executable $PROCESSES