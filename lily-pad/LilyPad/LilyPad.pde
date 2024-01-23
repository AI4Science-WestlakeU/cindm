/*********************************************************
                  Main Window!

Click the "Run" button to Run the simulation.

Change the geometry, flow conditions, numerical parameters
visualizations and measurements from this window.

This screen has an example. Other examples are found at 
the top of each tab. Copy/paste them here to run, but you 
can only have one setup & run at a time.

*********************************************************/

import java.util.Random;
BodyUnion bodyunion;
BDIM flow;
Body body1;
Body body2;
FloodPlot flood;
//SaveVectorFieldForEllipse data;
SaveVectorFieldFromBoundary data;
DiscNACA foil;
float t = 0., u = 0.;
int iter = 0, max_iter = 5;
float stime = 300., etime = 400.;
String lines;  // Array to store lines from the text file
String config_lines;
float[][] configs;
float[][] points;
float[] point;
float x;
float y;

void setup(){
  size(700,700);                             // display window size
}

float[][] parse_string(String lines) {
  
  // Remove square brackets and spaces
  lines = lines.replaceAll("\\[|\\]|\\s", "");
  // Split the string into individual elements based on commas
  String[] elements = split(lines, ',');
  // Determine the number of rows and columns in the array
  int rows = elements.length / 3; // Each point has three values: x, y, z
  int cols = 3; // Assuming three columns for x, y, z

  // Create a 2D array of floats
  points = new float[rows][cols];
  
  // Convert the elements to floats and populate the array
  int index = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      points[i][j] = float(elements[index]);
      index++;
    }
  }

  return points;
}

float[][] parse_string_config(String lines) {
  
  // Remove square brackets and spaces
  lines = lines.replaceAll("\\[|\\]|\\s", "");
  // Split the string into individual elements based on commas
  String[] elements = split(lines, ',');
  // Determine the number of rows and columns in the array
  int rows = elements.length / 9; // Each point has three values: x, y, z
  int cols = 9; // Assuming three columns for x, y, z

  // Create a 2D array of floats
  points = new float[rows][cols];
  
  // Convert the elements to floats and populate the array
  int index = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      points[i][j] = float(elements[index]);
      index++;
    }
  }

  return points;
}

// String input_path = "/Users/user/data/design_evaluation/designed_boundaries/";
// String output_sim_path = "/Users/user/data/design_evaluation/sim/";
// String output_force_path = "/Users/user/data/design_evaluation/force/";
// String output_boundary_path = "/Users/user/data/design_evaluation/reproduced_boundaries/";

//String input_path = "/Users/user/data/design_evaluation_check/designed_boundaries_check/";
//String output_sim_path = "/Users/user/data/design_evaluation_check/sim/";
//String output_force_path = "/Users/user/data/design_evaluation_check/force/";
//String output_boundary_path = "/Users/user/data/design_evaluation_check/reproduced_boundaries/";

String input_path = "./boundary_multiple/";
String output_sim_path = "./reporduced_simulation/sim/";
String output_force_path = "./reproduced_force/";
String output_boundary_path = "./reproduced_boundaries/";

void customsetup(int iteration){  
  // Create a Random object
  size(700,700); 
  int n=(int)pow(2,6); 
  Window view = new Window(n,n);
  
  config_lines = loadStrings("./config/sim_" + str(iteration) + ".txt")[0];
  configs = parse_string_config(config_lines);
  
  if(configs[0][8] == 0.){
    body1 = new EllipseBody(configs[0][0], configs[0][1], configs[0][2], configs[0][3], view); // define geom
    body1.rotate(configs[0][4]);     
  }
  else {
    body1 = new DiscNACA(configs[0][0], configs[0][1], configs[0][2], configs[0][3], view); // define geom
    body1.rotate(configs[0][4]);
  }
  if (configs[1][8] == 0.) {
    body2 = new EllipseBody(configs[0][0], configs[0][1], configs[0][2], configs[0][3], view); // define geom
    body2.rotate(configs[0][4]);     
  } 
  else {
    body2 = new DiscNACA(configs[1][0], configs[1][1], configs[1][2], configs[1][3], view); // define geom
    body2.rotate(configs[1][4]);
  } 
  
  bodyunion = new BodyUnion(body1, body2);
  flow = new BDIM(n,n,1.,bodyunion);             // solve for flow using BDIM
  flood = new FloodPlot(view);               // initialize a flood plot...
  flood.setLegend("vorticity",-.5,.5);       //    and its legend
  
  data = new SaveVectorFieldFromBoundary(
    output_sim_path + "sim_" + str(iteration) + ".txt", 
    output_force_path + "sim_" + str(iteration) + ".txt", 
    64, 
    64
  );

}
void draw(){
  if ((t == 0.) && (iter < max_iter)){
    customsetup(iter);
  }
  if(t<stime){  // run simulation until t<Time
    bodyunion.follow();                             // update the body
    flow.update(bodyunion); flow.update2();         // 2-step fluid update
    flood.display(flow.u.curl());              // compute and display vorticity
    bodyunion.display();      
    t+=flow.dt;
    //System.out.println(t);
    //System.out.println(flow.dt);
  }else if((stime<= t) && (t<etime)){  // run simulation until t<Time
    bodyunion.follow();                             // update the body
    flow.update(bodyunion); flow.update2();         // 2-step fluid update
    flood.display(flow.u.curl());              // compute and display vorticity
    bodyunion.display();      
    data.addField(flow.u, flow.p);
    data.addForce(bodyunion, flow.p);

    if (stime == t){
      for (int i = 0; i < 2; i++) {
        PrintWriter output;
        output = createWriter(output_boundary_path + "sim_" + str(iter) + "/boundary_" + str(i) + ".txt");
        output.print(bodyunion.bodyList.get(i).coords);
        output.flush();                           // Writes the remaining data to the file
        output.close();                           // Closes the file
      }    
    }
    t+=flow.dt;
  }
  else{  // close and save everything when t>Time
    data.finish();
    t = 0.;
    iter+=1;
    //exit();
  }
  if (max_iter <= iter){  // close and save everything when t>Time
    exit();
  }
}

/*
import java.util.Random;

BodyUnion bodyunion;
BDIM flow;
Body body;
Body body2;
Body body3;

FloodPlot flood;
SaveVectorFieldForEllipse data;
DiscNACA foil;
float t = 0., u = 0.;
//int iter = 0, max_iter = 200;
int iter = 0, max_iter = 5;
float stime = 300., etime = 500.;

void setup(){
  size(700,700);                             // display window size
}

void customsetup(int iteration){  
  // Create a Random object
  Random random = new Random();
  int boundarylowerBound = 0;
  int boundaryupperBound = 1;
  int boundarynum = boundarylowerBound + random.nextInt(boundaryupperBound - boundarylowerBound + 1);
  int boundarynum2 = boundarylowerBound + random.nextInt(boundaryupperBound - boundarylowerBound + 1);
  //int boundarynum3 = boundarylowerBound + random.nextInt(boundaryupperBound - boundarylowerBound + 1);
  //System.out.println("Random integer: " + xrandomInteger);
  //int boundarynum = 1;
  //int boundarynum2 = 1;

  int x0 = 1;
  int y0 = 1;
  int h0 = 1;
  int a0 = 1;
  float pipot = 0.1;
  int n0=(int)pow(2,6);
  data = new SaveVectorFieldForEllipse("saved/naca_ellipse_train_"+str(iteration)+".txt", x0, y0, h0, a0, pipot, n0, n0, iteration);    

  if (boundarynum == 0) {
    size(700,700); 
    int n=(int)pow(2,6); 
    float L = n/4., l = 0.2;                   // length-scale in grid units
    Window view = new Window(n,n);
        
    ////// 1st body ///////
    // Specify the range of random numbers you want to generate
    float xlowerBound = -0f;
    float xupperBound = 5f;
    float xrandomFloat = xlowerBound + random.nextFloat() * (xupperBound - xlowerBound + 1);
    //System.out.println("Random integer: " + xrandomInteger);
    
    float ylowerBound = -5f;
    float yupperBound = 5f;
    float yrandomFloat = ylowerBound + random.nextFloat() * (yupperBound - ylowerBound + 1);
    
    float hlowerBound = 0.4;
    float hupperBound = 1f;
    float hrandomFloat = hlowerBound + random.nextFloat() * (hupperBound - hlowerBound);
    
    float alowerBound = 2f;
    float aupperBound = 5f;
    float arandomFloat = alowerBound + random.nextFloat() * (aupperBound - alowerBound);
    
    float rotlowerBound = -1f;
    float rotupperBound = 1f;
    float rotrandomFloat = rotlowerBound + random.nextFloat() * (rotupperBound - rotlowerBound);
    
    float x = n/4 + xrandomFloat, y = n/2 + yrandomFloat;
    float h = L*l*hrandomFloat, a = l*arandomFloat;
    body = new EllipseBody(x,y,h,a,view); // define geom
    body.rotate(rotrandomFloat);
    float pivot=0.5;
    
    data.saveConfig(x, y, h, a, rotrandomFloat, pivot, n, n, 0.);
  }
  else if (boundarynum == 1) {
    size(700,700);                             // display window size
    int n=(int)pow(2,6);                       // number of grid points
    float l = 0.2;      
    Window view = new Window(n,n);

    // Specify the range of random numbers you want to generate
    float xlowerBound = -0f;
    float xupperBound = 5f;
    float xrandomFloat = xlowerBound + random.nextFloat() * (xupperBound - xlowerBound + 1);
    //System.out.println("Random integer: " + xrandomInteger);
    
    float ylowerBound = -5f;
    float yupperBound = 5f;
    float yrandomFloat = ylowerBound + random.nextFloat() * (yupperBound - ylowerBound + 1);
    
    float hlowerBound = -1.5f;
    float hupperBound = 1.5f;
    float hrandomFloat = hlowerBound + random.nextFloat() * (hupperBound - hlowerBound);
    
    float alowerBound = -0.05;
    float aupperBound = 0.15;
    float arandomFloat = alowerBound + random.nextFloat() * (aupperBound - alowerBound);
    
    float rotlowerBound = -1f;
    float rotupperBound = 1f;
    float rotrandomFloat = rotlowerBound + random.nextFloat() * (rotupperBound - rotlowerBound);
    //float rotrandomFloat = -0.3;

    float x = n/4 + xrandomFloat, y = n/2 + yrandomFloat;
    float h = 7. + hrandomFloat, a = l + arandomFloat;         // length-scale in grid units    
  
    body = new DiscNACA(x,y,h,a, view);
    body.rotate(rotrandomFloat);
    float pivot=0.5;
    
    data.saveConfig(x, y, h, a, rotrandomFloat, pivot, n, n, 1.);
  }

  if (boundarynum2 == 0) {
    size(700,700); 
    int n=(int)pow(2,6); 
    float L = n/4., l = 0.2;                   // length-scale in grid units
    Window view = new Window(n,n);
        
    ////// 2nd body ///////
    // Specify the range of random numbers you want to generate
    float xlowerBound2 = -0f;
    float xupperBound2 = 5f;
    float xrandomFloat2 = xlowerBound2 + random.nextFloat() * (xupperBound2 - xlowerBound2 + 1);
    //System.out.println("Random integer: " + xrandomInteger);
    
    float ylowerBound2 = -5f;
    float yupperBound2 = 5f;
    float yrandomFloat2 = ylowerBound2 + random.nextFloat() * (yupperBound2 - ylowerBound2 + 1);
    
    float hlowerBound2 = 0.4;
    float hupperBound2 = 1f;
    float hrandomFloat2 = hlowerBound2 + random.nextFloat() * (hupperBound2 - hlowerBound2);
    
    float alowerBound2 = 2f;
    float aupperBound2 = 5f;
    float arandomFloat2 = alowerBound2 + random.nextFloat() * (aupperBound2 - alowerBound2);
    
    float rotlowerBound2 = -1f;
    float rotupperBound2 = 1f;
    float rotrandomFloat2 = rotlowerBound2 + random.nextFloat() * (rotupperBound2 - rotlowerBound2);
    
    float x2 = n/2 + xrandomFloat2, y2 = n/3 + yrandomFloat2;
    float h2 = L*l*hrandomFloat2, a2 = l*arandomFloat2;
    body2 = new EllipseBody(x2,y2,h2,a2,view); // define geom
    body2.rotate(rotrandomFloat2);
    bodyunion = new BodyUnion(body, body2);
    
    flow = new BDIM(n,n,1.,bodyunion);             // solve for flow using BDIM
    flood = new FloodPlot(view);               // initialize a flood plot...
    flood.setLegend("vorticity",-.5,.5);       //    and its legend

    float pivot=0.5;
    
    data.saveConfig(x2, y2, h2, a2, rotrandomFloat2, pivot, n, n, 0.);
  }
  else if (boundarynum2 == 1) {
    size(700,700);                             // display window size
    int n=(int)pow(2,6);                       // number of grid points
    float l = 0.2;      
    Window view = new Window(n,n);

    // Specify the range of random numbers you want to generate
    float xlowerBound2 = -0f;
    float xupperBound2 = 5f;
    float xrandomFloat2 = xlowerBound2 + random.nextFloat() * (xupperBound2 - xlowerBound2 + 1);
    //System.out.println("Random integer: " + xrandomInteger);
    
    float ylowerBound2 = -5f;
    float yupperBound2 = 2f;
    float yrandomFloat2 = ylowerBound2 + random.nextFloat() * (yupperBound2 - ylowerBound2 + 1);
  
    float hlowerBound2 = -1.5f;
    float hupperBound2 = 1.5f;
    float hrandomFloat2 = hlowerBound2 + random.nextFloat() * (hupperBound2 - hlowerBound2);
  
    float alowerBound2 = -0.05;
    float aupperBound2 = 0.15;
    float arandomFloat2 = alowerBound2 + random.nextFloat() * (aupperBound2 - alowerBound2);

    float rotlowerBound2 = -1f;
    float rotupperBound2 = 1f;
    float rotrandomFloat2 = rotlowerBound2 + random.nextFloat() * (rotupperBound2 - rotlowerBound2);
    //float rotrandomFloat2 = 0.1;

    float x2 = n/2 + xrandomFloat2, y2 = n/3 + yrandomFloat2;
    float h2 = 7. + hrandomFloat2, a2 = l + arandomFloat2;         // length-scale in grid units    
  
    body2 = new DiscNACA(x2,y2,h2,a2, view);
    body2.rotate(rotrandomFloat2);
    bodyunion = new BodyUnion(body, body2);
    
    flow = new BDIM(n,n,1.,bodyunion);             // solve for flow using BDIM
    flood = new FloodPlot(view);               // initialize a flood plot...
    flood.setLegend("vorticity",-.5,.5);       //    and its legend

    float pivot=0.5;    
    data.saveConfig(x2, y2, h2, a2, rotrandomFloat2, pivot, n, n, 1.);
  }
*/
/*
  if (boundarynum3 == 0) {
    size(700,700); 
    int n=(int)pow(2,6); 
    float L = n/4., l = 0.2;                   // length-scale in grid units
    Window view = new Window(n,n);
    
    ////// 3rd body ///////
    // Specify the range of random numbers you want to generate
    float xlowerBound3 = -0f;
    float xupperBound3 = 5f;
    float xrandomFloat3 = xlowerBound3 + random.nextFloat() * (xupperBound3 - xlowerBound3 + 1);
    //System.out.println("Random integer: " + xrandomInteger);
    
    float ylowerBound3 = -5f;
    float yupperBound3 = 5f;
    float yrandomFloat3 = ylowerBound3 + random.nextFloat() * (yupperBound3 - ylowerBound3 + 1);
    
    float hlowerBound3 = 0.4;
    float hupperBound3 = 1f;
    float hrandomFloat3 = hlowerBound3 + random.nextFloat() * (hupperBound3 - hlowerBound3);
    
    float alowerBound3 = 2f;
    float aupperBound3 = 5f;
    float arandomFloat3 = alowerBound3 + random.nextFloat() * (aupperBound3 - alowerBound3);
    
    float rotlowerBound3 = -1f;
    float rotupperBound3 = 1f;
    float rotrandomFloat3 = rotlowerBound3 + random.nextFloat() * (rotupperBound3 - rotlowerBound3);
    
    float x3 = n/2 + xrandomFloat3, y3 = 2*n/3 + yrandomFloat3;
    float h3 = L*l*hrandomFloat3, a3 = l*arandomFloat3;
    body3 = new EllipseBody(x3,y3,h3,a3,view); // define geom
    body3.rotate(rotrandomFloat3);
    
    bodyunion = new BodyUnion(body, body2);
    bodyunion.add(body3);
    
    flow = new BDIM(n,n,1.,bodyunion);               // solve for flow using BDIM
    flood = new FloodPlot(view);                // intialize a flood plot...
    flood.setLegend("vorticity",-.5,.5);        //    and its legend
    
    float pivot=0.5;
    data.saveConfig(x3, y3, h3, a3, rotrandomFloat3, pivot, n, n, 0.);
 
  }
  else if (boundarynum3 == 1) {
    size(700,700);                             // display window size
    int n=(int)pow(2,6);                       // number of grid points
    float l = 0.2;      
    Window view = new Window(n,n);

    // Specify the range of random numbers you want to generate
    float xlowerBound3 = -0f;
    float xupperBound3 = 5f;
    float xrandomFloat3 = xlowerBound3 + random.nextFloat() * (xupperBound3 - xlowerBound3 + 1);
    //System.out.println("Random integer: " + xrandomInteger);
    
    float ylowerBound3 = -2f;
    float yupperBound3 = 5f;
    float yrandomFloat3 = ylowerBound3 + random.nextFloat() * (yupperBound3 - ylowerBound3 + 1);
  
    float hlowerBound3 = -1.5f;
    float hupperBound3 = 1.5f;
    float hrandomFloat3 = hlowerBound3 + random.nextFloat() * (hupperBound3 - hlowerBound3);
  
    float alowerBound3 = -0.05;
    float aupperBound3 = 0.15;
    float arandomFloat3 = alowerBound3 + random.nextFloat() * (aupperBound3 - alowerBound3);

    float rotlowerBound3 = -1f;
    float rotupperBound3 = 1f;
    float rotrandomFloat3 = rotlowerBound3 + random.nextFloat() * (rotupperBound3 - rotlowerBound3);

    float x3 = n/2 + xrandomFloat3, y3 = 2*n/3 + yrandomFloat3;
    float h3 = 7. + hrandomFloat3, a3 = l + arandomFloat3;         // length-scale in grid units    
  
    body3 = new DiscNACA(x3,y3,h3,a3, view);
    body3.rotate(rotrandomFloat3);
    
    bodyunion = new BodyUnion(body, body2);
    bodyunion.add(body3);    
    
    flow = new BDIM(n,n,1.,bodyunion);             // solve for flow using BDIM
    flood = new FloodPlot(view);               // initialize a flood plot...
    flood.setLegend("vorticity",-.5,.5);       //    and its legend

    float pivot=0.5;
    data.saveConfig(x3, y3, h3, a3, rotrandomFloat3, pivot, n, n, 1.);  }
*/
/*
}
void draw(){
  if ((t == 0.) && (iter < max_iter)){
    customsetup(iter);
  }
  if(t<stime){  // run simulation until t<Time
    bodyunion.follow();                             // update the body
    flow.update(bodyunion); flow.update2();         // 2-step fluid update
    flood.display(flow.u.curl());              // compute and display vorticity
    bodyunion.display();      
    t+=flow.dt;
    //System.out.println(t);
    //System.out.println(flow.dt);
  }else if((stime<= t) && (t<etime)){  // run simulation until t<Time
    bodyunion.follow();                             // update the body
    flow.update(bodyunion); flow.update2();         // 2-step fluid update
    flood.display(flow.u.curl());              // compute and display vorticity
    bodyunion.display();      
    data.addField(flow.u, flow.p);
    data.addForce(bodyunion, flow.p);
    if (stime == t){
      for (int i = 0; i < bodyunion.bodyList.size(); i++) {
        PrintWriter output;
        output = createWriter("boundary_multiple/sim_"+str(iter)+"/boundary_"+str(i)+".txt");
        output.print(bodyunion.bodyList.get(i).coords);
        output.flush();                           // Writes the remaining data to the file
        output.close();                           // Closes the file
      }    
    }
    t+=flow.dt;
  }
  else{  // close and save everything when t>Time
    data.finish();
    t = 0.;
    iter+=1;
    //exit();
  }
  if (max_iter <= iter){  // close and save everything when t>Time
    exit();
  }
}
*/

/*
BodyUnion body;
Body singlebody;
Body singlebody1;
BDIM flow;
FloodPlot flood;

void setup(){
  int n=(int)pow(2,6);
  size(700,700);
  Window view = new Window(n,n);

  singlebody = new NACA(35,20,14,0.2,view);
  singlebody.rotate(0.05);

  singlebody1 = new NACA(30,40,14,0.2,view);
  singlebody1.rotate(-0.09);

  body = new BodyUnion( singlebody, singlebody1);
  
  singlebody1 = new NACA(20,30,14,0.2,view);
  singlebody1.rotate(-0.1);
  
  body.add(singlebody1);

  //body.add(new NACA(30,35,12,0.2,view));
  //body.add(new NACA(20,50,12, 0.2, view));
  //body.rotate(0.01);
  
  flow = new BDIM(n,n,1.,body);             // solve for flow using BDIM
  flood = new FloodPlot(view);               // initialize a flood plot...
  flood.setLegend("vorticity",-.5,.5);       //    and its legend

}
void draw(){
  background(0);
  body.follow(); // uncomment to move as a group
  flow.update(body); flow.update2();         // 2-step fluid update
  flood.display(flow.u.curl());              // compute and display vorticity
  //for (Body child : body.bodyList) child.follow(); // uncomment to move individually
  body.display();
}
//void mousePressed(){body.mousePressed();}
//void mouseReleased(){body.mouseReleased();}
*/

/*
import java.util.Random;

BDIM flow;
Body body;
FloodPlot flood;
//SaveVectorFieldForEllipse data;
SaveVectorFieldFromBoundary data;
DiscNACA foil;
float t = 0., u = 0.;
int iter = 0, max_iter = 10;
float stime = 300., etime = 400.;
String lines;  // Array to store lines from the text file
float[][] points;
float[] point;
float x;
float y;

void setup(){
  size(700,700);                             // display window size
}

float[][] parse_string(String lines) {
  
  // Remove square brackets and spaces
  lines = lines.replaceAll("\\[|\\]|\\s", "");
  // Split the string into individual elements based on commas
  String[] elements = split(lines, ',');
  // Determine the number of rows and columns in the array
  int rows = elements.length / 3; // Each point has three values: x, y, z
  int cols = 3; // Assuming three columns for x, y, z

  // Create a 2D array of floats
  points = new float[rows][cols];
  
  // Convert the elements to floats and populate the array
  int index = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      points[i][j] = float(elements[index]);
      index++;
    }
  }
  // Print the array for verification
  //for (int i = 0; i < rows; i++) {
  //  for (int j = 0; j < 2; j++) {
  //    print(points[i][j] + "\t");
  //  }
  //  println();
  //}
 
  return points;
}

void customsetup(int iteration){  
  // Create a Random object
  size(700,700); 
  int n=(int)pow(2,6); 
  Window view = new Window(n,n);

  lines = loadStrings("/Users/user/data/boundary_test2/sim_0/boundary_"+str(iteration)+".txt")[0];
  points = parse_string(lines);
  println("points: ", points);
  x = points[0][0]; 
  y = points[0][1];
    
  body = new GENERATED_NACA(x,y, points, view);
  
  //for (int i = 0; i < points.length; i++) {
  //  x = points[i][0]; 
  //  y = points[i][1];
  //  println(i, x, y);
  //  PVector vector = new PVector(x, y);
  //  body.coords.add(vector);
  //  //body.coords[i][0] = float(lines[i][0]);
  //  //body.coords[i][1] = float(lines[i][1]);
  //}
  
  flow = new BDIM(n,n,1.,body);             // solve for flow using BDIM
  flood = new FloodPlot(view);               // initialize a flood plot...
  flood.setLegend("vorticity",-.5,.5);       //    and its legend
  
  data = new SaveVectorFieldFromBoundary("/Users/user/data/design/simulation_"+str(iteration)+".txt", 64, 64);

}
void draw(){
  if ((t == 0.) && (iter < max_iter)){
    customsetup(iter);
  }
  if(t<stime){  // run simulation until t<Time
    body.follow();                             // update the body
    flow.update(body); flow.update2();         // 2-step fluid update
    flood.display(flow.u.curl());              // compute and display vorticity
    body.display();      
    t+=flow.dt;
    //System.out.println(t);
    //System.out.println(flow.dt);
  }else if((stime<= t) && (t<etime)){  // run simulation until t<Time
    body.follow();                             // update the body
    flow.update(body); flow.update2();         // 2-step fluid update
    flood.display(flow.u.curl());              // compute and display vorticity
    body.display();      
    data.addField(flow.u, flow.p);
    PrintWriter output;
    output = createWriter("/Users/user/data/design/sim_"+str(iter)+"/boundary_"+str(int(t-stime))+".txt");
    output.print(body.coords);
    output.flush();                           // Writes the remaining data to the file
    output.close();                           // Closes the file
    t+=flow.dt;
  }
  else{  // close and save everything when t>Time
    data.finish();
    t = 0.;
    iter+=1;
    //exit();
  }
  if (max_iter <= iter){  // close and save everything when t>Time
    exit();
  }
}
*/

/*
import java.util.Random;

BDIM flow;
Body body;
FloodPlot flood;
SaveVectorFieldForEllipse data;
int example = 3; // Choose an example reaction function
float t = 0., u = 0.;
int iter = 0, max_iter = 100;

void setup(){
  size(700,700);                             // display window size
}
void customsetup(int iteration){
  // Create a Random object
  Random random = new Random();
  // Specify the range of random numbers you want to generate
  int xlowerBound = 0;
  int xupperBound = 20;
  // Generate a random integer within the specified range
  int xrandomInteger = xlowerBound + random.nextInt(xupperBound - xlowerBound + 1);
  // Output the random integer
  //System.out.println("Random integer: " + xrandomInteger);
  
  int ylowerBound = -5;
  int yupperBound = 5;
  int yrandomInteger = ylowerBound + random.nextInt(yupperBound - ylowerBound + 1);

  float hlowerBound = -5f;
  float hupperBound = 0f;
  float hrandomFloat = hlowerBound + random.nextFloat() * (hupperBound - hlowerBound);

  float alowerBound = -0.01;
  float aupperBound = 0.4;
  float arandomFloat = alowerBound + random.nextFloat() * (aupperBound - alowerBound);

  size(700,700); 
  int n=(int)pow(2,7);  
  float L = n/4., l = 0.2;                   // length-scale in grid units
  float x = n/3+xrandomInteger, y = n/2+yrandomInteger;
  float h = L*l+hrandomFloat, a = l+arandomFloat, pivot = 0.5;
  Window view = new Window(n,n);
  body = new ChaoticEllipse(x, y, h, a, pivot,example,view); // define geom
  flow = new BDIM(n,n,1.,body);               // solve for flow using BDIM
  flood = new FloodPlot(view);                // intialize a flood plot...
  flood.setLegend("vorticity",-.5,.5);        //    and its legend
  data = new SaveVectorFieldForEllipse("saved/ellipse_"+str(iteration)+".txt", x, y, h, a, pivot, n, n);
}
void draw(){
  if ((t == 0.) && (iter < max_iter)){
    customsetup(iter);
    System.out.println("Iteration: " + iter);
  }
  if(t<100.){  // run simulation until t<Time
    body.react(flow);
    flow.update(body); flow.update2();         // 2-step fluid update
    flood.display(flow.u.curl());              // compute and display vorticity
    body.display(); // display the body
    t+=flow.dt;
    //System.out.println(t);
    //System.out.println(flow.dt);
  }else if((100.<= t) && (t<400.)){  // run simulation until t<Time
    body.react(flow);
    flow.update(body); flow.update2();         // 2-step fluid update
    data.addField(flow.u, flow.p);
    flood.display(flow.u.curl());              // compute and display vorticity
    body.display();                            // display the body
    PrintWriter output;
    output = createWriter("boundary/sim_"+str(iter)+"/boundary_"+str(int(t)-100)+".txt");
    output.print(body.coords);
    output.flush();                           // Writes the remaining data to the file
    output.close();                           // Closes the file
    t+=flow.dt;
  }
  else{  // close and save everything when t>Time
    data.finish();
    t = 0.;
    iter+=1;
    //exit();
  }
  if (max_iter <= iter){  // close and save everything when t>Time
    exit();
  }
}
*/

/*
import java.util.Random;

BDIM flow;
Body body;
FloodPlot flood;
SaveVectorFieldForEllipse data;
int example = 3; // Choose an example reaction function
float t = 0.;
String sim_id = "0";

void setup(){
  size(700,700);                             // display window size
  int n=(int)pow(2,7);                       // number of grid points
  float L = n/4., l = 0.2;                   // length-scale in grid units
  float x = n/3+10, y = n/2-5;
  float h = L*l+2, a = l, pivot = 0.5;
  Window view = new Window(n,n);
  body = new ChaoticEllipse(x, y, h, a, pivot,example,view); // define geom
  flow = new BDIM(n,n,1.,body);               // solve for flow using BDIM
  flood = new FloodPlot(view);                // intialize a flood plot...
  flood.setLegend("vorticity",-.5,.5);        //    and its legend
  data = new SaveVectorFieldForEllipse("saved/ellipse_00004.txt", x, y, h, a, pivot, n, n);
}
void draw(){
  if(t<100.){  // run simulation until t<Time
    body.react(flow);
    flow.update(body); flow.update2();         // 2-step fluid update
    flood.display(flow.u.curl());              // compute and display vorticity
    body.display(); // display the body
    t+=flow.dt;
  }else if((100.<= t) && (t<400.)){  // run simulation until t<Time
    body.react(flow);
    flow.update(body); flow.update2();         // 2-step fluid update
    data.addField(flow.u, flow.p);
    flood.display(flow.u.curl());              // compute and display vorticity
    body.display(); // display the body
    PrintWriter output;
    output = createWriter("boundary/sim_000003/boundary_"+str(int(t)-100)+".txt");
    output.print(body.coords);
    output.flush(); // Writes the remaining data to the file
    output.close(); // Closes the file
    t+=flow.dt;
  }
  else{  // close and save everything when t>Time
    data.finish();

    exit();
  }
}
*/

/*
BDIM flow;
Body body;
FloodPlot flood;
SaveVectorFieldForEllipse eldata;
int example = 3; // Choose an example reaction function
float t = 0.;

void setup(){
  size(700,700);                             // display window size
  int n=(int)pow(2,7);                       // number of grid points
  float L = n/4., l = 0.2;                   // length-scale in grid units
  float x = n/3+10, y = n/2-5;
  float h = L*l+2, a = l, pivot = 0.5;
  Window view = new Window(n,n);
  body = new ChaoticEllipse(x, y, h, a, pivot,example,view); // define geom
  flow = new BDIM(n,n,1.,body);               // solve for flow using BDIM
  flood = new FloodPlot(view);                // intialize a flood plot...
  flood.setLegend("vorticity",-.5,.5);        //    and its legend
  eldata = new SaveVectorFieldForEllipse("saved/ellipse_00003.txt", x, y, h, a, pivot, n, n);
}

void draw(){
  if(t<100.){  // run simulation until t<Time
    body.react(flow);
    flow.update(body); flow.update2();         // 2-step fluid update
    //data.addField(flow.u, flow.p);
    flood.display(flow.u.curl());              // compute and display vorticity
    body.display(); // display the body
    t+=flow.dt;
  }else if((100.<= t) && (t<400.)){  // run simulation until t<Time
    body.react(flow);
    flow.update(body); flow.update2();         // 2-step fluid update
    eldata.addField(flow.u, flow.p);
    flood.display(flow.u.curl());              // compute and display vorticity
    body.display(); // display the body
    PrintWriter output;
    output = createWriter("boundary/sim_000003/boundary_"+str(int(t)-100)+".txt");
    output.print(body.coords);
    output.flush(); // Writes the remaining data to the file
    output.close(); // Closes the file
    t+=flow.dt;
  }
  else{  // close and save everything when t>Time
    eldata.finish();
    exit();
  }
}
*/
/*
// Circle that can be dragged by the mouse
BDIM flow;
Body body;
FloodPlot flood;

void setup(){
  size(700,700);                             // display window size
  int n=(int)pow(2,7);                       // number of grid points
  float L = n/8.;                            // length-scale in grid units
  Window view = new Window(n,n);

  body = new CircleBody(n/3,n/2,L,view);     // define geom
  flow = new BDIM(n,n,1.5,body);             // solve for flow using BDIM
  flood = new FloodPlot(view);               // initialize a flood plot...
  flood.setLegend("vorticity",-.5,.5);       //    and its legend
}
void draw(){
  body.follow();                             // update the body
  flow.update(body); flow.update2();         // 2-step fluid update
  flood.display(flow.u.curl());              // compute and display vorticity
  body.display();                            // display the body
}
void mousePressed(){body.mousePressed();}    // user mouse...
void mouseReleased(){body.mouseReleased();}  // interaction methods
void mouseWheel(MouseEvent event){body.mouseWheel(event);}
*/
