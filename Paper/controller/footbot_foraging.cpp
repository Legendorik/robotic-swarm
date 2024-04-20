/* Include the controller definition */
#include "footbot_foraging.h"
/* Function definitions for XML parsing */
#include <argos3/core/utility/configuration/argos_configuration.h>
/* 2D vector definition */
#include <argos3/core/utility/math/vector2.h>
/* Logging */
#include <argos3/core/utility/logging/argos_log.h>
#include <argos3/core/simulator/simulator.h>
#include <argos3/core/simulator/loop_functions.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <boost/algorithm/string.hpp>



/****************************************/
/****************************************/

CFootBotForaging::SFoodData::SFoodData() :
   HasFoodItem(false),
   FoodItemIdx(0),
   TotalFoodItems(0) {}

void CFootBotForaging::SFoodData::Reset() {
   HasFoodItem = false;
   FoodItemIdx = 0;
   TotalFoodItems = 0;
}

/****************************************/
/****************************************/

CFootBotForaging::SDiffusionParams::SDiffusionParams() :
   GoStraightAngleRange(CRadians(-1.0f), CRadians(1.0f)) {}

void CFootBotForaging::SDiffusionParams::Init(TConfigurationNode& t_node) {
   try {
      CRange<CDegrees> cGoStraightAngleRangeDegrees(CDegrees(-10.0f), CDegrees(10.0f));
      GetNodeAttribute(t_node, "go_straight_angle_range", cGoStraightAngleRangeDegrees);
      GoStraightAngleRange.Set(ToRadians(cGoStraightAngleRangeDegrees.GetMin()),
                               ToRadians(cGoStraightAngleRangeDegrees.GetMax()));
      GetNodeAttribute(t_node, "delta", Delta);
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error initializing controller diffusion parameters.", ex);
   }
}

/****************************************/
/****************************************/

void CFootBotForaging::SWheelTurningParams::Init(TConfigurationNode& t_node) {
   try {
      TurningMechanism = NO_TURN;
      CDegrees cAngle;
      GetNodeAttribute(t_node, "hard_turn_angle_threshold", cAngle);
      HardTurnOnAngleThreshold = ToRadians(cAngle);
      GetNodeAttribute(t_node, "soft_turn_angle_threshold", cAngle);
      SoftTurnOnAngleThreshold = ToRadians(cAngle);
      GetNodeAttribute(t_node, "no_turn_angle_threshold", cAngle);
      NoTurnAngleThreshold = ToRadians(cAngle);
      GetNodeAttribute(t_node, "max_speed", MaxSpeed);
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error initializing controller wheel turning parameters.", ex);
   }
}

/****************************************/
/****************************************/

CFootBotForaging::SStateData::SStateData() :
   ProbRange(0.0f, 1.0f) {}

void CFootBotForaging::SStateData::Init(TConfigurationNode& t_node) {
   try {
      GetNodeAttribute(t_node, "initial_rest_to_explore_prob", InitialRestToExploreProb);
      GetNodeAttribute(t_node, "initial_explore_to_rest_prob", InitialExploreToRestProb);
      GetNodeAttribute(t_node, "food_rule_explore_to_rest_delta_prob", FoodRuleExploreToRestDeltaProb);
      GetNodeAttribute(t_node, "food_rule_rest_to_explore_delta_prob", FoodRuleRestToExploreDeltaProb);
      GetNodeAttribute(t_node, "collision_rule_explore_to_rest_delta_prob", CollisionRuleExploreToRestDeltaProb);
      GetNodeAttribute(t_node, "social_rule_rest_to_explore_delta_prob", SocialRuleRestToExploreDeltaProb);
      GetNodeAttribute(t_node, "social_rule_explore_to_rest_delta_prob", SocialRuleExploreToRestDeltaProb);
      GetNodeAttribute(t_node, "minimum_resting_time", MinimumRestingTime);
      GetNodeAttribute(t_node, "minimum_unsuccessful_explore_time", MinimumUnsuccessfulExploreTime);
      GetNodeAttribute(t_node, "minimum_search_for_place_in_nest_time", MinimumSearchForPlaceInNestTime);

   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error initializing controller state parameters.", ex);
   }
}

void CFootBotForaging::SStateData::Reset() {
   State = STATE_RESTING;
   InNest = true;
   RestToExploreProb = InitialRestToExploreProb;
   ExploreToRestProb = InitialExploreToRestProb;
   TimeExploringUnsuccessfully = 0;
   /* Initially the robot is resting, and by setting RestingTime to
      MinimumRestingTime we force the robots to make a decision at the
      experiment start. If instead we set RestingTime to zero, we would
      have to wait till RestingTime reaches MinimumRestingTime before
      something happens, which is just a waste of time. */
   TimeRested = MinimumRestingTime;
   TimeSearchingForPlaceInNest = 0;
   mGlobalX = 0;
   mGlobalY = 0;
}

/****************************************/
/****************************************/

CFootBotForaging::CFootBotForaging() :
   m_pcWheels(NULL),
   m_pcLEDs(NULL),
   m_pcRABA(NULL),
   m_pcRABS(NULL),
   m_pcProximity(NULL),
   m_pcLight(NULL),
   m_pcGround(NULL),
   m_pcDistanceA(NULL),
   m_pcDistanceS(NULL),
   m_pcCamera(NULL),
   m_pcRNG(NULL) {}

/****************************************/
/****************************************/

void CFootBotForaging::Init(TConfigurationNode& t_node) {
   try {
      /*
       * Initialize sensors/actuators
       */
      m_pcWheels    = GetActuator<CCI_DifferentialSteeringActuator>("differential_steering");
      m_pcLEDs      = GetActuator<CCI_LEDsActuator                >("leds"                 );
      m_pcRABA      = GetActuator<CCI_RangeAndBearingActuator     >("range_and_bearing"    );
      m_pcDistanceA = GetActuator<CCI_FootBotDistanceScannerActuator>("footbot_distance_scanner");
      m_pcRABS      = GetSensor  <CCI_RangeAndBearingSensor       >("range_and_bearing"    );
      m_pcProximity = GetSensor  <CCI_FootBotProximitySensor      >("footbot_proximity"    );
      m_pcLight     = GetSensor  <CCI_FootBotLightSensor          >("footbot_light"        );
      m_pcGround    = GetSensor  <CCI_FootBotMotorGroundSensor    >("footbot_motor_ground" );
      m_pcPos       = GetSensor  <CCI_PositioningSensor           >("positioning"          );
      m_pcDistanceS = GetSensor  <CCI_FootBotDistanceScannerSensor>("footbot_distance_scanner");
      m_pcCamera = GetSensor  <CCI_ColoredBlobOmnidirectionalCameraSensor>("colored_blob_omnidirectional_camera");
      /*
       * Parse XML parameters
       */
      /* Diffusion algorithm */
      m_sDiffusionParams.Init(GetNode(t_node, "diffusion"));
      /* Wheel turning */
      m_sWheelTurningParams.Init(GetNode(t_node, "wheel_turning"));
      /* Controller state */
      m_sStateData.Init(GetNode(t_node, "state"));
      // m_pcDistanceA->Enable();
      // m_pcDistanceS->Enable();
      m_pcCamera->Enable();
      // m_pcCamera->GetReadings();
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error initializing the foot-bot foraging controller for robot \"" << GetId() << "\"", ex);
   }
   /*
    * Initialize other stuff
    */
   /* Create a random number generator. We use the 'argos' category so
      that creation, reset, seeding and cleanup are managed by ARGoS. */
   m_pcRNG = CRandom::CreateRNG("argos");

   std::string full_id = std::string(GetId().c_str(), 5);
   m_id = std::stoi(full_id.erase(0, 2));

   std::cout << "FILES_ID: " << getenv("FILES_ID") << std::endl;
   files_id = getenv("FILES_ID");
   if (files_id == NULL) {
    files_id = "0";
   }
   createMappings();
   Reset();
}

void CFootBotForaging::Destroy() {
  munmap(data_mmap, file_size);
  munmap(actions_mmap, file_size);
}

/****************************************/
/****************************************/

void CFootBotForaging::createMappings(){
  
  char data_filename[25];
  char actions_filename[25];
  snprintf(data_filename, 25, "%s%s%d", files_id, data_fname, m_id);
  snprintf(actions_filename, 25, "%s%s%d",files_id, actions_fname, m_id);
  int data_fd = -1;
  if ((data_fd = open(data_filename, O_RDWR, 0)) == -1){
    std::cerr << "Unable to open" << data_filename << std::endl;
  }
  int actions_fd = -1;
  if ((actions_fd = open(actions_filename, O_RDWR, 0)) == -1){
    std::cerr << "Unable to open" << actions_filename << std::endl;
  }

  // open the file in shared memory
  data_mmap = (char*) mmap(NULL, file_size, PROT_WRITE, MAP_SHARED, data_fd, 0);
  actions_mmap = (char*) mmap(NULL, file_size, PROT_READ, MAP_SHARED, actions_fd, 0);
  close(data_fd);
  close(actions_fd);
}

void CFootBotForaging::doSend(char data[max_length], std::size_t length){

  char buf[file_size];
  strncpy(buf, data, length);
  for (size_t i = length; i < file_size; i++)
  {
    buf[i] = '\0';
  }
  if (iter % 15 == 0) {}
   //  std::cout<< m_id << " SEND: " << buf << std::endl;
  
  strcpy(data_mmap, buf);
}

/****************************************/
/****************************************/

void CFootBotForaging::doReceive(){

  char buf[file_size];
  strncpy(buf, actions_mmap, file_size);
  std::vector<std::string> result;
  boost::split(result, buf, boost::is_any_of(";"));
  auto time = std::time(nullptr);
//   std::cout << std::ctime(&time) << " " << m_id << " " << last_received_iter  << " RECEIVE: $" << buf << "$" << std::endl;
  if (!result.empty() && !(result.size() == 1 && result[0].empty())) {
   
    if (last_received_iter != result[0]) {
      last_received_iter = result[0];

      // std::cout<< m_id << " RECEIVE: " << buf << std::endl;
      if (result.size() >= 3) {
         SetWheelSpeedsFromVector(CVector2(std::stof(result[1]), std::stof(result[2])));
      }
      else if (result.size() >= 2) {
         std::cout<< m_id << " RECEIVE: " << buf << std::endl;
         if (result[1] == "RESET") {
            if (m_id == 0) {
               CSimulator &cSimulator = CSimulator::GetInstance();
               cSimulator.Reset();
            }
            else {
               Reset();
            }

         }
      }
      
    }
    else {
      // sched_yield();
      nanosleep((const struct timespec[]){{0, 1000000L}}, NULL); //100000L
      // std::cout<< m_id <<" YIELD "<< buf << std::endl;
      // if (result[1] == "RESET") {
         
      // } else {
         doReceive();
      // }
      
    }

  }
  
}

/****************************************/
/****************************************/

std::string CFootBotForaging::GetPackage(){
   
   std::stringstream package;
   package << std::fixed << std::setprecision(3);
   package << iter << ";" // message id
         << m_sStateData.mGlobalX << ";" << m_sStateData.mGlobalY << ";" << m_sStateData.mRotationZ << ";" // global position
         << m_sStateData.InNest << ";" // is robot in nest
         << m_sFoodData.HasFoodItem << ";"; // is robot carrying food

   auto groundReadings = m_pcGround->GetReadings();
   auto rabReadings = m_pcRABS->GetReadings();
   auto cameraReadings = m_pcCamera->GetReadings();

   auto lightVector = CalculateTrueVectorToLight();
   package << lightVector.GetX() << ";" << lightVector.GetY() << ";";
   auto proxVector = CalculateProximityVector();
   package << proxVector.GetX() << ";" << proxVector.GetY() << ";";
   auto isCollision = CFootBotForaging::isCollision(proxVector);
   package << isCollision << ";";

   auto rabReadingSize = rabReadings.size();
   // send RAB sensor data
   package << rabReadingSize << ";";
   for (size_t i = 0; i < rabReadingSize; i++)
   {
      auto rabVector = CVector2(rabReadings[i].Range / 100, rabReadings[i].HorizontalBearing);
      // package << rabReadings[i].Range / 100 << ";";
      // package << rabReadings[i].HorizontalBearing.GetValue() << ";";

      package << rabVector.GetX() << ";";
      package << rabVector.GetY() << ";";

      // is robot have food
      package << rabReadings[i].Data[0] << ";";
      // is robot see food
      package << rabReadings[i].Data[1] << ";";
   }

   auto cameraReadingSize = cameraReadings.BlobList.size();

   auto cameraReadingBlueSize = 0;
   for (size_t i = 0; i < cameraReadingSize; i++) {
      if (cameraReadings.BlobList[i]->Color.GetBlue() == 255) {
         cameraReadingBlueSize++;
      }
   }
   // // send camera sensor data
   package << cameraReadingBlueSize << ";";
   for (size_t i = 0; i < cameraReadingSize; i++)
   {
      if (cameraReadings.BlobList[i]->Color.GetBlue() == 255) {
         auto cameraVector = CVector2(cameraReadings.BlobList[i]->Distance / 100, cameraReadings.BlobList[i]->Angle);
         // package << cameraReadings.BlobList[i]->Distance / 100 << ";";
         // package << cameraReadings.BlobList[i]->Angle.GetValue() << ";";
         package << cameraVector.GetX() << ";";
         package << cameraVector.GetY() << ";";
         package << cameraReadings.BlobList[i]->Color.GetRed() << ";";
         package << cameraReadings.BlobList[i]->Color.GetGreen() << ";";
         package << cameraReadings.BlobList[i]->Color.GetBlue() << ";";
      }

   }

  return package.str();

}
/****************************************/
/****************************************/

void CFootBotForaging::ControlStep() {
   iter++;
   doReceive();


   const CCI_PositioningSensor::SReading& sPosRead = m_pcPos->GetReading();
   CRadians cZAngle, cYAngle, cXAngle;
   sPosRead.Orientation.ToEulerAngles(cZAngle, cYAngle, cXAngle);        
   m_sStateData.Update(sPosRead.Position.GetX(), sPosRead.Position.GetY(), cZAngle.SignedNormalize().GetValue());

   setRABData();
   UpdateState();

   // switch(m_sStateData.State) {
   //    case SStateData::STATE_RESTING: {
   //       Rest();
   //       break;
   //    }
   //    case SStateData::STATE_EXPLORING: {
   //       Explore();
   //       break;
   //    }
   //    case SStateData::STATE_RETURN_TO_NEST: {
   //       ReturnToNest();
   //       break;
   //    }
   //    default: {
   //       LOGERR << "We can't be here, there's a bug!" << std::endl;
   //    }
   // }
   std::string msg = GetPackage();
   // std::cout << "package size "  << msg.size() << std::endl;
   char pack[msg.size() + 1];
   strcpy(pack, msg.c_str());

   doSend(pack, sizeof(pack));
   // sched_yield();
}

void CFootBotForaging::setRABData() {

   m_pcRABA->SetData(0, m_sFoodData.HasFoodItem);
   auto cameraReadings = m_pcCamera->GetReadings();
   bool seeFood = false;

   for (size_t i = 0; i < cameraReadings.BlobList.size(); i++)
   {
      auto blob = cameraReadings.BlobList[i];
      if (blob->Color.GetBlue() == 255 && blob->Color.GetGreen() == 0 && blob->Color.GetRed() == 0) {
         seeFood = true;
         break;
      }
   }
   m_pcRABA->SetData(1, seeFood);


}

/****************************************/
/****************************************/
void CFootBotForaging::Reset() {
   // std::cout<< m_id << " RESET!!! " << std::endl;
   doSend("0;", 2);
   /* Reset robot state */
   m_sStateData.Reset();
   /* Reset food data */
   m_sFoodData.Reset();
   iter = 0;
   /* Set LED color */
   // m_pcLEDs->SetAllColors(CColor::RED);
   m_pcLEDs->SetAllColors(CColor::GREEN);
   /* Clear up the last exploration result */
   m_eLastExplorationResult = LAST_EXPLORATION_NONE;
   m_pcRABA->ClearData();
   // m_pcRABA->SetData(0, LAST_EXPLORATION_NONE);
}

/****************************************/
/****************************************/

void CFootBotForaging::UpdateState() {
   /* Reset state flags */
   m_sStateData.InNest = false;
   /* Read stuff from the ground sensor */
   const CCI_FootBotMotorGroundSensor::TReadings& tGroundReads = m_pcGround->GetReadings();
   /*
    * You can say whether you are in the nest by checking the ground sensor
    * placed close to the wheel motors. It returns a value between 0 and 1.
    * It is 1 when the robot is on a white area, it is 0 when the robot
    * is on a black area and it is around 0.5 when the robot is on a gray
    * area. 
    * The foot-bot has 4 sensors like this, two in the front
    * (corresponding to readings 0 and 1) and two in the back
    * (corresponding to reading 2 and 3).  Here we want the back sensors
    * (readings 2 and 3) to tell us whether we are on gray: if so, the
    * robot is completely in the nest, otherwise it's outside.
    */
   if(tGroundReads[2].Value > 0.25f &&
      tGroundReads[2].Value < 0.75f &&
      tGroundReads[3].Value > 0.25f &&
      tGroundReads[3].Value < 0.75f) {
      m_sStateData.InNest = true;
   }
}

/****************************************/
/****************************************/

CVector2 CFootBotForaging::CalculateVectorToLight() {
   /* Get readings from light sensor */
   const CCI_FootBotLightSensor::TReadings& tLightReads = m_pcLight->GetReadings();
   /* Sum them together */
   CVector2 cAccumulator;
   for(size_t i = 0; i < tLightReads.size(); ++i) {
      cAccumulator += CVector2(tLightReads[i].Value, tLightReads[i].Angle);
   }
   /* If the light was perceived, return the vector */
   if(cAccumulator.Length() > 0.0f) {
      return CVector2(1.0f, cAccumulator.Angle());
   }
   /* Otherwise, return zero */
   else {
      return CVector2();
   }
}

CVector2 CFootBotForaging::CalculateTrueVectorToLight() {
   /* Get readings from light sensor */
   const CCI_FootBotLightSensor::TReadings& tLightReads = m_pcLight->GetReadings();
   /* Sum them together */
   CVector2 cAccumulator;
   for(size_t i = 0; i < tLightReads.size(); ++i) {
      cAccumulator += CVector2(tLightReads[i].Value, tLightReads[i].Angle);
   }
   /* If the light was perceived, return the vector */
   if(cAccumulator.Length() > 0.0f) {
      return CVector2(cAccumulator.Length(), cAccumulator.Angle());
   }
   /* Otherwise, return zero */
   else {
      return CVector2();
   }
}

CVector2 CFootBotForaging::CalculateProximityVector() {
   /* Get readings from proximity sensor */
   const CCI_FootBotProximitySensor::TReadings& tProxReads = m_pcProximity->GetReadings();
   /* Sum them together */
   CVector2 cDiffusionVector = CVector2();
   for(size_t i = 0; i < tProxReads.size(); ++i) {
      cDiffusionVector += CVector2(tProxReads[i].Value, tProxReads[i].Angle);
   }

   return cDiffusionVector;
}

bool CFootBotForaging::isCollision(CVector2 proxVector) {
   if (proxVector.Length() > 0.99) {
      return true;
   }
   return false;
}
/****************************************/
/****************************************/

CVector2 CFootBotForaging::DiffusionVector(bool& b_collision) {
   /* Get readings from proximity sensor */
   const CCI_FootBotProximitySensor::TReadings& tProxReads = m_pcProximity->GetReadings();
   /* Sum them together */
   CVector2 cDiffusionVector;
   for(size_t i = 0; i < tProxReads.size(); ++i) {
      cDiffusionVector += CVector2(tProxReads[i].Value, tProxReads[i].Angle);
   }
   /* If the angle of the vector is small enough and the closest obstacle
      is far enough, ignore the vector and go straight, otherwise return
      it */
   if(m_sDiffusionParams.GoStraightAngleRange.WithinMinBoundIncludedMaxBoundIncluded(cDiffusionVector.Angle()) &&
      cDiffusionVector.Length() < m_sDiffusionParams.Delta ) {
      b_collision = false;
      return CVector2::X;
   }
   else {
      b_collision = true;
      cDiffusionVector.Normalize();
      return -cDiffusionVector;
   }
}

/****************************************/
/****************************************/

void CFootBotForaging::SetWheelSpeedsFromVector(const CVector2& c_heading) {
   /* Get the heading angle */
   CRadians cHeadingAngle = c_heading.Angle().SignedNormalize();
   /* Get the length of the heading vector */
   Real fHeadingLength = c_heading.Length();
   /* Clamp the speed so that it's not greater than MaxSpeed */
   Real fBaseAngularWheelSpeed = Min<Real>(fHeadingLength, m_sWheelTurningParams.MaxSpeed);
   /* State transition logic */
   if(m_sWheelTurningParams.TurningMechanism == SWheelTurningParams::HARD_TURN) {
      if(Abs(cHeadingAngle) <= m_sWheelTurningParams.SoftTurnOnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::SOFT_TURN;
      }
   }
   if(m_sWheelTurningParams.TurningMechanism == SWheelTurningParams::SOFT_TURN) {
      if(Abs(cHeadingAngle) > m_sWheelTurningParams.HardTurnOnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::HARD_TURN;
      }
      else if(Abs(cHeadingAngle) <= m_sWheelTurningParams.NoTurnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::NO_TURN;
      }
   }
   if(m_sWheelTurningParams.TurningMechanism == SWheelTurningParams::NO_TURN) {
      if(Abs(cHeadingAngle) > m_sWheelTurningParams.HardTurnOnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::HARD_TURN;
      }
      else if(Abs(cHeadingAngle) > m_sWheelTurningParams.NoTurnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::SOFT_TURN;
      }
   }
   /* Wheel speeds based on current turning state */
   Real fSpeed1, fSpeed2;
   switch(m_sWheelTurningParams.TurningMechanism) {
      case SWheelTurningParams::NO_TURN: {
         /* Just go straight */
         fSpeed1 = fBaseAngularWheelSpeed;
         fSpeed2 = fBaseAngularWheelSpeed;
         break;
      }
      case SWheelTurningParams::SOFT_TURN: {
         /* Both wheels go straight, but one is faster than the other */
         Real fSpeedFactor = (m_sWheelTurningParams.HardTurnOnAngleThreshold - Abs(cHeadingAngle)) / m_sWheelTurningParams.HardTurnOnAngleThreshold;
         fSpeed1 = fBaseAngularWheelSpeed - fBaseAngularWheelSpeed * (1.0 - fSpeedFactor);
         fSpeed2 = fBaseAngularWheelSpeed + fBaseAngularWheelSpeed * (1.0 - fSpeedFactor);
         break;
      }
      case SWheelTurningParams::HARD_TURN: {
         /* Opposite wheel speeds */
         fSpeed1 = -m_sWheelTurningParams.MaxSpeed;
         fSpeed2 =  m_sWheelTurningParams.MaxSpeed;
         break;
      }
   }
   /* Apply the calculated speeds to the appropriate wheels */
   Real fLeftWheelSpeed, fRightWheelSpeed;
   if(cHeadingAngle > CRadians::ZERO) {
      /* Turn Left */
      fLeftWheelSpeed  = fSpeed1;
      fRightWheelSpeed = fSpeed2;
   }
   else {
      /* Turn Right */
      fLeftWheelSpeed  = fSpeed2;
      fRightWheelSpeed = fSpeed1;
   }
   /* Finally, set the wheel speeds */
   m_pcWheels->SetLinearVelocity(fLeftWheelSpeed, fRightWheelSpeed);
}

/****************************************/
/****************************************/

void CFootBotForaging::Rest() {
   /* If we have stayed here enough, probabilistically switch to
    * 'exploring' */
   if(m_sStateData.TimeRested > m_sStateData.MinimumRestingTime &&
      m_pcRNG->Uniform(m_sStateData.ProbRange) < m_sStateData.RestToExploreProb) {
      m_pcLEDs->SetAllColors(CColor::GREEN);
      m_sStateData.State = SStateData::STATE_EXPLORING;
      m_sStateData.TimeRested = 0;
   }
   else {
      ++m_sStateData.TimeRested;
      /* Be sure not to send the last exploration result multiple times */
      if(m_sStateData.TimeRested == 1) {
         m_pcRABA->SetData(0, LAST_EXPLORATION_NONE);
      }
      /*
       * Social rule: listen to what other people have found and modify
       * probabilities accordingly
       */
      const CCI_RangeAndBearingSensor::TReadings& tPackets = m_pcRABS->GetReadings();
      for(size_t i = 0; i < tPackets.size(); ++i) {
         switch(tPackets[i].Data[0]) {
            case LAST_EXPLORATION_SUCCESSFUL: {
               m_sStateData.RestToExploreProb += m_sStateData.SocialRuleRestToExploreDeltaProb;
               m_sStateData.ProbRange.TruncValue(m_sStateData.RestToExploreProb);
               m_sStateData.ExploreToRestProb -= m_sStateData.SocialRuleExploreToRestDeltaProb;
               m_sStateData.ProbRange.TruncValue(m_sStateData.ExploreToRestProb);
               break;
            }
            case LAST_EXPLORATION_UNSUCCESSFUL: {
               m_sStateData.ExploreToRestProb += m_sStateData.SocialRuleExploreToRestDeltaProb;
               m_sStateData.ProbRange.TruncValue(m_sStateData.ExploreToRestProb);
               m_sStateData.RestToExploreProb -= m_sStateData.SocialRuleRestToExploreDeltaProb;
               m_sStateData.ProbRange.TruncValue(m_sStateData.RestToExploreProb);
               break;
            }
         }
      }
   }
}

/****************************************/
/****************************************/

void CFootBotForaging::Explore() {
   /* We switch to 'return to nest' in two situations:
    * 1. if we have a food item
    * 2. if we have not found a food item for some time;
    *    in this case, the switch is probabilistic
    */
   bool bReturnToNest(false);
   /*
    * Test the first condition: have we found a food item?
    * NOTE: the food data is updated by the loop functions, so
    * here we just need to read it
    */
   if(m_sFoodData.HasFoodItem) {
      /* Apply the food rule, decreasing ExploreToRestProb and increasing
       * RestToExploreProb */
      m_sStateData.ExploreToRestProb -= m_sStateData.FoodRuleExploreToRestDeltaProb;
      m_sStateData.ProbRange.TruncValue(m_sStateData.ExploreToRestProb);
      m_sStateData.RestToExploreProb += m_sStateData.FoodRuleRestToExploreDeltaProb;
      m_sStateData.ProbRange.TruncValue(m_sStateData.RestToExploreProb);
      /* Store the result of the expedition */
      m_eLastExplorationResult = LAST_EXPLORATION_SUCCESSFUL;
      /* Switch to 'return to nest' */
      bReturnToNest = true;
   }
   /* Test the second condition: we probabilistically switch to 'return to
    * nest' if we have been wandering for some time and found nothing */
   else if(m_sStateData.TimeExploringUnsuccessfully > m_sStateData.MinimumUnsuccessfulExploreTime) {
      if (m_pcRNG->Uniform(m_sStateData.ProbRange) < m_sStateData.ExploreToRestProb) {
         /* Store the result of the expedition */
         m_eLastExplorationResult = LAST_EXPLORATION_UNSUCCESSFUL;
         /* Switch to 'return to nest' */
         bReturnToNest = true;
      }
      else {
         /* Apply the food rule, increasing ExploreToRestProb and
          * decreasing RestToExploreProb */
         m_sStateData.ExploreToRestProb += m_sStateData.FoodRuleExploreToRestDeltaProb;
         m_sStateData.ProbRange.TruncValue(m_sStateData.ExploreToRestProb);
         m_sStateData.RestToExploreProb -= m_sStateData.FoodRuleRestToExploreDeltaProb;
         m_sStateData.ProbRange.TruncValue(m_sStateData.RestToExploreProb);
      }
   }
   /* So, do we return to the nest now? */
   if(bReturnToNest) {
      /* Yes, we do! */
      m_sStateData.TimeExploringUnsuccessfully = 0;
      m_sStateData.TimeSearchingForPlaceInNest = 0;
      m_pcLEDs->SetAllColors(CColor::BLUE);
      m_sStateData.State = SStateData::STATE_RETURN_TO_NEST;
   }
   else {
      /* No, perform the actual exploration */
      ++m_sStateData.TimeExploringUnsuccessfully;
      UpdateState();
      /* Get the diffusion vector to perform obstacle avoidance */
      bool bCollision;
      CVector2 cDiffusion = DiffusionVector(bCollision);
      /* Apply the collision rule, if a collision avoidance happened */
      if(bCollision) {
         /* Collision avoidance happened, increase ExploreToRestProb and
          * decrease RestToExploreProb */
         m_sStateData.ExploreToRestProb += m_sStateData.CollisionRuleExploreToRestDeltaProb;
         m_sStateData.ProbRange.TruncValue(m_sStateData.ExploreToRestProb);
         m_sStateData.RestToExploreProb -= m_sStateData.CollisionRuleExploreToRestDeltaProb;
         m_sStateData.ProbRange.TruncValue(m_sStateData.RestToExploreProb);
      }
      /*
       * If we are in the nest, we combine antiphototaxis with obstacle
       * avoidance
       * Outside the nest, we just use the diffusion vector
       */
      if(m_sStateData.InNest) {
         /*
          * The vector returned by CalculateVectorToLight() points to
          * the light. Thus, the minus sign is because we want to go away
          * from the light.
          */
         SetWheelSpeedsFromVector(
            m_sWheelTurningParams.MaxSpeed * cDiffusion -
            m_sWheelTurningParams.MaxSpeed * 0.25f * CalculateVectorToLight());
      }
      else {
         /* Use the diffusion vector only */
         SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * cDiffusion);
      }
   }
}

/****************************************/
/****************************************/

void CFootBotForaging::ReturnToNest() {
   /* As soon as you get to the nest, switch to 'resting' */
   UpdateState();
   /* Are we in the nest? */
   if(m_sStateData.InNest) {
      /* Have we looked for a place long enough? */
      if(m_sStateData.TimeSearchingForPlaceInNest > m_sStateData.MinimumSearchForPlaceInNestTime) {
         /* Yes, stop the wheels... */
         m_pcWheels->SetLinearVelocity(0.0f, 0.0f);
         /* Tell people about the last exploration attempt */
         m_pcRABA->SetData(0, m_eLastExplorationResult);
         /* ... and switch to state 'resting' */
         m_pcLEDs->SetAllColors(CColor::RED);
         m_sStateData.State = SStateData::STATE_RESTING;
         m_sStateData.TimeSearchingForPlaceInNest = 0;
         m_eLastExplorationResult = LAST_EXPLORATION_NONE;
         return;
      }
      else {
         /* No, keep looking */
         ++m_sStateData.TimeSearchingForPlaceInNest;
      }
   }
   else {
      /* Still outside the nest */
      m_sStateData.TimeSearchingForPlaceInNest = 0;
   }
   /* Keep going */
   bool bCollision;
   SetWheelSpeedsFromVector(
      m_sWheelTurningParams.MaxSpeed * DiffusionVector(bCollision) +
      m_sWheelTurningParams.MaxSpeed * CalculateVectorToLight());
}

/****************************************/
/****************************************/

/*
 * This statement notifies ARGoS of the existence of the controller.
 * It binds the class passed as first argument to the string passed as
 * second argument.
 * The string is then usable in the XML configuration file to refer to
 * this controller.
 * When ARGoS reads that string in the XML file, it knows which controller
 * class to instantiate.
 * See also the XML configuration files for an example of how this is used.
 */
REGISTER_CONTROLLER(CFootBotForaging, "footbot_foraging_controller")
