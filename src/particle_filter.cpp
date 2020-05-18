/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <random>
#include "multiv_gauss.cpp"
#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  if (!initialized()) {
   	//Set the number of particles
    num_particles = 100;  

    //Create normal distributions for x, y, and theta
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    //Define random data generator
    std::default_random_engine gen;

    for(int i=0;i<num_particles;++i){
      particles.emplace_back(Particle{i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0});
    }
    weights.insert(weights.begin(),num_particles,1.0);
    //Set filter to initialized state
    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  double x_new_no_noise, y_new_no_noise, theta_new_no_noise;
  
  //Define random data generator
  std::default_random_engine gen;
  
  for(auto p = particles.begin(); p != particles.end(); ++p){
    //Check if yaw rate is equal to 0
    if (yaw_rate != 0.0){
        x_new_no_noise = (*p).x + (velocity/yaw_rate)*(sin((*p).theta + (yaw_rate*delta_t)) - sin((*p).theta));
        y_new_no_noise = (*p).y + (velocity/yaw_rate)*(cos((*p).theta) - cos((*p).theta + (yaw_rate*delta_t)));
    }
    else {
        x_new_no_noise = (*p).x + velocity*cos((*p).theta)*delta_t;
        y_new_no_noise = (*p).y + velocity*sin((*p).theta)*delta_t;
    }
    theta_new_no_noise = (*p).theta + (yaw_rate*delta_t);
  
    //Create normal distributions for x, y, and theta
    std::normal_distribution<double> dist_x(x_new_no_noise, std_pos[0]);
    std::normal_distribution<double> dist_y(y_new_no_noise, std_pos[1]);
    std::normal_distribution<double> dist_theta(theta_new_no_noise, std_pos[2]);

    //Add noise to the new state from gaussian distributions
    (*p).x = dist_x(gen);
    (*p).y = dist_y(gen);
    (*p).theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  //Associate landmark in map coordinate system
  
  for(auto obs = observations.begin(); obs != observations.end(); ++obs){
    int closest_id = -1;
    double closest_dist = std::numeric_limits<double>::infinity();
    for(auto pred = predicted.begin(); pred != predicted.end(); ++pred){
        double current_dist = dist((*pred).x,(*pred).y,(*obs).x,(*obs).y);
    	if(current_dist < closest_dist){
          closest_id = (*pred).id;
          closest_dist = current_dist;
        }
    }
    (*obs).id = closest_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  for(auto p = particles.begin(); p != particles.end(); ++p){
    
    //1) Filter viewed map landmarks
    vector<LandmarkObs> predicted_landmarks;
	for(auto l = map_landmarks.landmark_list.begin(); l != map_landmarks.landmark_list.end(); ++l){
      if(dist((*p).x,(*p).y,(*l).x_f,(*l).y_f) <= sensor_range) {
        predicted_landmarks.push_back(LandmarkObs {(*l).id_i, (*l).x_f, (*l).y_f});
      }
    }
    
    //2) Transform observations from vehicle coordinate system to map coordinate system
    vector<LandmarkObs> transformed_observations;
    for(auto o = observations.begin(); o != observations.end(); ++o){
        int id = (*o).id;
      	double t_x = (*p).x + (cos((*p).theta)*(*o).x) - (sin((*p).theta)*(*o).y);
        double t_y = (*p).y + (sin((*p).theta)*(*o).x) + (cos((*p).theta)*(*o).y);
    	transformed_observations.push_back(LandmarkObs {id,t_x,t_y});
    }

    //3) Associate observations to landmarks
    dataAssociation(predicted_landmarks,transformed_observations);

    //4) Particles weights calculation from association
    for(auto t_o = transformed_observations.begin(); t_o != transformed_observations.end(); ++t_o){
    	for(auto p_l = predicted_landmarks.begin(); p_l != predicted_landmarks.end(); ++p_l){
          if((*t_o).id == (*p_l).id){
            (*p).weight *= multiv_prob(std_landmark[0],std_landmark[1],(*t_o).x,(*t_o).y,(*p_l).x,(*p_l).y);
          }
        }
    }
   
  }
  
  //5) Particles weights normalization
  double total_weight = 0.0;
  for(auto p = particles.begin(); p != particles.end(); ++p) total_weight += (*p).weight;
  for (int i = 0; i < particles.size(); ++i) {
    particles[i].weight /= total_weight;
    weights[i] = particles[i].weight;
    if(i<10) std::cout << weights[i] << "-";
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  //Filtered set of particles
  std::vector<Particle> filtered_particles;
  
  //Resampling wheel algorithm
  
  //Uniform distributions
  std::default_random_engine generator;
  std::uniform_int_distribution<int> random_index(0,num_particles-1);
  std::uniform_real_distribution<double> random_weight(0,2*(*max_element(weights.begin(),weights.end())));
  
  //Initial conditions
  int index = random_index(generator);
  double beta = 0.0;
  
  for(int i=0; i<particles.size(); ++i){
    //Add to beta a random weight angle and find the corresponding particle
  	beta += random_weight(generator);
    while (weights[index] < beta){
    	beta -= weights[index];
        index = (index+1)%num_particles;
    }
    //Push selected particle ti filtered set
    filtered_particles.push_back(particles[index]);
  }
  
  particles = filtered_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}