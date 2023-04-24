#!/bin/bash
# script to generate a list of submit files and submit them to condor
EXEC=$1
runlist=$2
jobname=$3



# set up results directory
dir=$PWD/$jobname/runlist_`date '+%y%m%d_%H.%M.%S'`
echo "Setting up results directory: $dir"
mkdir $PWD/$jobname
mkdir $dir
# preamble
echo "
Executable = $EXEC
Requirements = Machine==\"ece017.ece.local.cmu.edu\" || Machine==\"ece018.ece.local.cmu.edu\"   || Machine==\"ece021.ece.local.cmu.edu\" || Machine==\"ece022.ece.local.cmu.edu\" || Machine==\"ece024.ece.local.cmu.edu\"
Should_Transfer_Files = IF_NEEDED
When_To_Transfer_Output = ON_EXIT
Notification = ERROR
Image_size = 100GB
Request_gpus = 1
Request_cpus = 14
InitialDir = $dir" > $dir/runlist.sub

while read p; do
  echo "$EXEC $p"
  echo "
  Arguments = $p
  Output = process_\$(cluster).\$(process).txt
  Error = process_\$(cluster).\$(process).err
  Log = process_\$(cluster).\$(process).log
  queue" >> $dir/runlist.sub
done <$runlist
#submit to condor
condor_submit $dir/runlist.sub