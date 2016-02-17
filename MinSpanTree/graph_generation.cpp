#include <iostream>
#include <vector>
#include <ctime>
#include <random>
#include <cmath>
#include <thread>

typedef struct Vertex {

    vector<double> coords;
    Vertex* parent;
    int rank;
} Vertex;

typedef struct Edge {

    Vertex* u;
    Vertex* v;
    double distance;
} Edge;

struct edgeCompare {
    bool operator() (Edge* e1, Edge* e2) { return (e1->distance < e1->distance);}
} edgeCompare;

typedef struct Graph {
    int V;
    int E;
    vector<Edge*> edges;
    vector<Vertex*> vertices;
} Graph;
