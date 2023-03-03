#!/bin/bash
# Workflow script for particle sim
# sh run_simulation.sh 50 35 8 for 50 particles for 35 timesteps for 8 nodes
#cd $DATA

#execute code
#/usr/bin/mpiexec.hydra -f /home/tsimo1/RHPC/bin/worker_list.txt -np 4 /home/tsimo1/RHPC/apps/nbodypipe $1 $2 
#mpirun -np 4 $1 $2

containers=($(docker ps | awk '{if(NR>1) print $NF}'))
for ((i = 0; i < ${#containers[@]}; ++i)); do
  echo "Fetching data from container: ${containers[$i]}"

  $(docker exec ${containers[$i]} /bin/sh -c "cp timedat.dat /tmp/")
  $(docker exec ${containers[$i]} /bin/sh -c "echo $NAME")
  $(docker cp ${containers[$i]}:/tmp/timedat.dat timedat.${i})
done

echo ================================

plot=""
i=0;
while (($i <= $3-1))
do
    plot+="\"timedat.$i\" i $i u 4:5 pt 3  ps 1 t \"Node $i\""

    if [[ $i < $(($3-1)) ]]; then
      plot+=", "
    else
      plot+=";"
    fi

    let i=$i+1
done

i=0;
while (($i <= $2-1))
do
    printf "set xrange [0:1]
    set title \"$1 particles at timestep $i\"
    set yrange [0:1]
    set grid
    set term gif size 640,768
    set output '$i.gif'
    plot " > data_$i.gnu
    j=0
    while (($j <= $3-1))
    do
      printf "\"timedat.$j\" i $i u 4:5 pt 3  ps 1 t \"Node $j\"" >> data_$i.gnu
      if [[ $j < $(($3-1)) ]]; then
        printf ", " >> data_$i.gnu
      else
        printf ";" >> data_$i.gnu
      fi
      let j=j+1
    done
    gnuplot data_$i.gnu
    let i=$i+1
done

rm animation.gif

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