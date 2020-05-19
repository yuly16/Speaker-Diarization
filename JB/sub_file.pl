#!/usr/bin/perl
$file_name=$ARGV[0];
$file_path=$ARGV[1];
$output_path=$ARGV[2];
$sub_file=$ARGV[3];
open(FILE1,"<$file_path/xvector.scp")||die('error readin');
open(FILE2,">$output_path/$sub_file")||die('error writin');
while(<FILE1>){
   @line=split(/[ _]/,$_);
   if($line[0] eq $file_name){
     print FILE2 $_;
   }
}
close(FILE1);
close(FILE2);

