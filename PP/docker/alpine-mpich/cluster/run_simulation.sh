#!/bin/bash
# Workflow script for particle sim
# sh run_simulation.sh 50 35 for 50 particles for 35 timesteps
#cd $DATA

#execute code
#/usr/bin/mpiexec.hydra -f /home/tsimo1/RHPC/bin/worker_list.txt -np 4 /home/tsimo1/RHPC/apps/nbodypipe $1 $2 
#mpirun -np 4 $1 $2

containers=($(docker ps | awk '{if(NR>1) print $NF}'))
for ((i = 0; i < ${#containers[@]}; ++i)); do
  echo "Container: ${containers[$i]}"
  echo "i: ${i}"

  $(docker exec ${containers[$i]} /bin/sh -c "cp timedat.dat /tmp/")
  $(docker exec ${containers[$i]} /bin/sh -c "echo $NAME")
  $(docker cp ${containers[$i]}:/tmp/timedat.dat timedat.${i})
  echo ================================
done

#performn visualization
i=0;
while (($i <= $2-1))
do
    echo "set xrange [0:1]
    set title \"$1 particles at timestep $i\"
    set yrange [0:1]
    set grid
    set term gif size 320,480
    set output '$i.gif'
    plot \"timedat.0\" i $i u 4:5 pt 3  ps 1 t \"Node 0\", \"timedat.1\" i $i u 4:5 pt 3  ps 1 t \"Node 1\",\"timedat.2\" i $i u 4:5 pt 3 ps 1 t \"Node 2\", \"timedat.3\" i $i u 4:5 pt 3  ps 1 t \"Node 3\";" >data_$i.gnu
    gnuplot data_$i.gnu
    let i=$i+1
done

# Generating animated file
echo "Generating animated file"
ls *.gif | sort -nk1 | xargs ./gifmerge -10 -l0 > animation.vid

# Cleanup
echo "Clearing temp files"
rm timedat.*
rm *.gnu
rm *.gif

mv animation.vid animation.gif

echo "Done! animation.gif generated!"