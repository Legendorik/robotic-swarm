#include "foraging_qt_user_functions.h"
#include "footbot_foraging.h"
#include <argos3/core/simulator/entity/controllable_entity.h>

using namespace argos;

/****************************************/
/****************************************/

CForagingQTUserFunctions::CForagingQTUserFunctions() {
   RegisterUserFunction<CForagingQTUserFunctions,CFootBotEntity>(&CForagingQTUserFunctions::Draw);
}

/****************************************/
/****************************************/

void CForagingQTUserFunctions::Draw(CFootBotEntity& c_entity) {
   CFootBotForaging& cController = dynamic_cast<CFootBotForaging&>(c_entity.GetControllableEntity().GetController());
   CFootBotForaging::SFoodData& sFoodData = cController.GetFoodData();
   if(sFoodData.HasFoodItem) {
      DrawCylinder(
         CVector3(0.0f, 0.0f, 0.3f), 
         CQuaternion(),
         0.1f,
         0.05f,
         CColor::BLACK);
   }
   // auto lightVector = cController.CalculateTrueVectorToLight();
   // DrawRay(CRay3(CVector3(0, 0, 0.3), CVector3(lightVector.GetX(), lightVector.GetY(), 0.3)));
}

/****************************************/
/****************************************/

REGISTER_QTOPENGL_USER_FUNCTIONS(CForagingQTUserFunctions, "foraging_qt_user_functions")
