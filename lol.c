#include <stdio.h>
#include <stdlib.h>

void ler(int *array, int n){
	int i;
	for(i = 0; i < n; i++){
		scanf("%d",&array[i]);
	}
}

void print(int *arr, int n){
	int i;
	for(i = 0;i < n;i++){
		printf("%d\n",arr[i]);
	}
}

void main(){
	int *array;
	int n = 4;
	array = malloc(n * sizeof(int));

	//printf("%d",sizeof(int));

	ler(array, n);

	print(array, n);

}